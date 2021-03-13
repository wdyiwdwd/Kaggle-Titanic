
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing 
import math


# In[2]:


#读取数据并且把所有数据结合到一起
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
passengerId=test_data['PassengerId']
all_data = pd.concat([train_data, test_data])
all_data.head()


# In[3]:


#对全部集合进行处理的函数
def  clean_data(df):
    # 替换称谓 
    df['Title'] = df['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False).str.strip()
    df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer' , inplace = True)
    df['Title'].replace(['Jonkheer','Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty' , inplace = True)
    df['Title'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs' , inplace = True)
    df['Title'].replace(['Mlle', 'Miss'], 'Miss' , inplace = True)
    df['Title'].replace(['Mr'], 'Mr' , inplace = True)
    df['Title'].replace(['Master'], 'Master' , inplace = True)
    
    #用随机森林预测年龄
    age_df = df[['Age', 'Pclass','Sex','Title']]
    age_df = pd.get_dummies(age_df)
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
#     #放大年龄的影响，增加孩子属性，年龄小于16岁均视为孩子
#     df['Child'] = df['Age'].apply(lambda x: 1 if x < 16 else 0)
    
    #Embarked用众数S代替空值
    df['Embarked'] = df['Embarked'].fillna('S')
    
    #Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3 用这个分组平均数填充
    fare_mean = df[(df['Embarked'] == "S") & (df['Pclass'] == 3)].Fare.mean()
    df['Fare']=df['Fare'].fillna(fare_mean)
    
    #对Cabin进行处理,提取首字母
    df['Cabin'] = df['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else 'U')
    
    #增加家庭规模属性
    df['FamilySize'] = df['SibSp'] + df['Parch']
    #在1到3范围内存活率最高赋值2 大于6存活率最低赋值0
    df['Family'] = df['FamilySize'].apply(lambda x: 0 if x > 6 else (2 if x >=1 and x <=3 else 1))
    
    
    #新增TicketGroup特征，统计每个乘客的共票号数。
    Ticket_Count = dict(df['Ticket'].value_counts())
    df['TicketGroup'] = df['Ticket'].apply(lambda x:Ticket_Count[x])
    #同Family
    df['TicketGroup'] = df['TicketGroup'].apply(lambda x: 0 if x > 8 else (2 if x >=2 and x <=4 else 1))
    
    #把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性
    df['Surname']=df['Name'].apply(lambda x:x.split(',')[0].strip())
    Surname_Count = dict(df['Surname'].value_counts())
    df['FamilyGroup'] = df['Surname'].apply(lambda x:Surname_Count[x])
    Female_Child_Group=df.loc[(df['Family']>=2) & ((df['Age']<=12) | (df['Sex']=='female'))]
    Male_Adult_Group=df.loc[(df['Family']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
    
    #因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组中存活率为1的设置为幸存组，推测处于遇难组的女性和儿童幸存的可能性较低，处于幸存组的成年男性幸存的可能性较高。
    Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
    Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
    Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
    Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

    #为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
    train=all_data.loc[all_data['Survived'].notnull()]
    test=all_data.loc[all_data['Survived'].isnull()]
    test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
    test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
    test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
    test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
    test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
    test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
    
    df = pd.concat([train, test])
    
        
    #对连续数据进行标准化处理
    scaler = preprocessing.StandardScaler()
    df['Fare'] = scaler.fit_transform(df.filter(['Fare']))
    df['Age'] = scaler.fit_transform(df.filter(['Age']))

    #drop掉无用属性
    df = df.drop(['PassengerId'], axis=1)
    df = df.drop(['Ticket'], axis=1)
    df = df.drop(['FamilySize'], axis=1)
    df = df.drop(['Parch'], axis=1)
    df = df.drop(['SibSp'], axis=1)
    df = df.drop(['Name'], axis=1)
    df = df.drop(['Surname'], axis=1)
    df = df.drop(['FamilyGroup'], axis=1)

    #独热编码
    df = pd.get_dummies(df)
    
    return df


# In[4]:


all_data = clean_data(all_data)
all_data.head()


# In[5]:


train=all_data[all_data['Survived'].notnull()]
 #不知道为什么 Survived变成了float 需要换成int
train['Survived'] = train['Survived'].apply(lambda x:int(x))
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
train_data_X = train.drop(['Survived'],axis=1)
train_data_Y = train['Survived']


# In[6]:


# pipe=Pipeline([('select',SelectKBest(k=20)), 
#                ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

# param_test = {'classify__n_estimators':list(range(20,50,2)), 
#               'classify__max_depth':list(range(3,60,3))}
# gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)
# gsearch.fit(train_data_X, train_data_Y)


# In[14]:


#交叉验证
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(train_data_X[:700], train_data_Y[:700])
predictions = pipeline.predict(train_data_X[700:])
fail_values = train_data_Y[700:] - predictions
accuracy = fail_values.value_counts()[0]/len(fail_values)
accuracy


# In[8]:


select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(train_data_X, train_data_Y)


# In[9]:


predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": passengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("result.csv", index=False)

