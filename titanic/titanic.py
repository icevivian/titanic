#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 17:00
# @Author  : Aries
# @Site    : 
# @File    : titanic.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import model_selection
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
pd.set_option('display.width',1000)
#pd.set_option('display.height',1000)
#pd.set_option('display.max_rows',500)
#pd.set_option('display.max_columns',500)
#pd.set_option('display.width',1000)

data_train=pd.read_csv('train.csv')
#print(data_train)
#data_train.info()
'''
分析数据，寻找特征
'''
'''
fig=plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar') #或者用plt.bar(x,y)
plt.title("获救情况(1为获救)",fontproperties=font_set)
plt.ylabel('人数',fontproperties=font_set)

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title("乘客等级分布",fontproperties=font_set)
plt.ylabel("人数",fontproperties=font_set)

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel('年龄',fontproperties=font_set)
plt.grid(b=True,which="major",axis='y')
plt.title("按年龄看获救分布（1为获救)",fontproperties=font_set)

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel("年龄",fontproperties=font_set)
plt.ylabel("密度",fontproperties=font_set)
plt.title("各等级的乘客年龄分布",fontproperties=font_set)
plt.legend(("1","2","3"),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind="bar")
plt.title("各登船口岸上船人数",fontproperties=font_set)
plt.ylabel("人数",fontproperties=font_set)
#plt.show()

#等级与获救的关系
Survived_0=data_train.Pclass[data_train.Survived==0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'获救':Survived_1,'unsurvived':Survived_0})
df.plot(kind='bar',stacked=True,)
plt.title('各乘客等级的获救情况',fontproperties=font_set)
plt.xlabel('等级',fontproperties=font_set)
plt.ylabel('人数',fontproperties=font_set)

#性别与获救的关系
Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'survived': Survived_1, 'unsurvived': Survived_0})
df.plot(kind='bar', stacked=True, )
plt.title('各性别的获救情况', fontproperties=font_set)
plt.xlabel('性别', fontproperties=font_set)
plt.ylabel('人数', fontproperties=font_set)

#等级and性别与获救的关系
fig=plt.figure()
fig.set(alpha=0.2)
plt.title('Pclass and Sex for survived')

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Pclass!=3][data_train.Sex=='female'].value_counts().plot(kind='bar',label='female,high class',color='red')
ax1.legend(['female,Pclass!=3'],loc='best')


ax2=fig.add_subplot(142,sharey=ax1)
data_train.Survived[data_train.Pclass==3][data_train.Sex=='female'].value_counts().plot(kind='bar',color='pink')
ax2.legend(['Pclass==3'],loc='best')

ax3=fig.add_subplot(143,sharey=ax1)
data_train.Survived[data_train.Pclass!=3][data_train.Sex=='male'].value_counts().plot(kind='bar',color='lightblue')
ax3.legend(['man,Pclass!=3'],loc='best')

ax4=fig.add_subplot(144,sharey=ax1)
data_train.Survived[data_train.Pclass==3][data_train.Sex=='male'].value_counts().plot(kind='bar',color='steelblue')
ax4.legend(['man,Pclass==3'],loc='best')
plt.show()

g=data_train.groupby(['SibSp','Survived'])
df=pd.DataFrame(g.count()['PassengerId'])
#print(df)
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.ensemble import BaggingRegressor
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"   #loc()方法：对行，列位置进行索引
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
     画出data在某模型上的learning curve.参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    #自己的理解：将所有样本分成cv份，训练样本数最大为总数的cv-1份，因此横坐标是从最小依次增加到最大，均分成20份。
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
            plt.xlabel(u"traning set size")
            plt.ylabel(u"score")
            plt.gca().invert_yaxis()
            plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                             alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                             alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"traning set")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cv set")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

if  __name__=='__main__':
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    #print(data_train)

    '''
    get_dummies()：是一种reshape功能，将定性特征转换为定量特征。
    para: prefix:reshape后参数命名前缀
    concat()：将dataframe结合起来的方法
    '''
    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    #print(df)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    #print(df)
    '''
    preprocessing数据预处理，归一化
    fit(X),X应该为二维数组，而df['Age']是一个一维的Series,所以先用values取Series中的数值，类型为ndarray，然后用reshape(-1,1)将一维数组转换为二维数组
    '''
    scaler=preprocessing.StandardScaler()
    age_scale_param=scaler.fit(df['Age'].values.reshape(-1,1))
    df['Age_scaled']=scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)
    fare_scale_param=scaler.fit(df['Fare'].values.reshape(-1,1))
    df['Fare_scaled']=scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)
    #print(df)

    '''逻辑回归模型拟合'''
    train_df=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np=train_df.as_matrix()
    y = train_np[:, 0]
    X=train_np[:,1:]
    clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
    clf.fit(X,y)

    '''测试模型效果'''
    #处理缺省数据：Fare,Age,Cabin
    data_test=pd.read_csv('test.csv')
    data_test.loc[(data_test.Fare.isnull()),'Fare']=0
    tmp_df=data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    tmp_mat= tmp_df[tmp_df.Age.isnull()].as_matrix()
    X_test=tmp_mat[:,1:]
    predictedAge=rfr.predict(X_test)
    data_test.loc[(data_test.Age.isnull()),'Age']=predictedAge
    data_test = set_Cabin_type(data_test)
    #特征因子化
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # print(df)
    #归一化数据
    df_test['Age_scaled']=scaler.fit_transform(df_test['Age'].values.reshape(-1,1),age_scale_param)
    df_test['Fare_scaled']=scaler.fit_transform(df_test['Fare'].values.reshape(-1,1),fare_scale_param)
    #print(df_test)

    test_df=df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions=clf.predict(test_df)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictions.csv", index=False)

    '''做好一个初始模型之后，还只是完成了第一步，接下来我们需要对这个模型进行分析，看它是过拟合还是欠拟合，以确定我们需要
    更多的特征还是更多的数据，或者是其他的操作。可以使用learning curve
    首先这里还存在两个问题：1.被丢弃的两个特征：Name和Ticket;2.年龄的拟合是否可靠
    首先看各个特征与模型之间的相关性是否与我们观察到的相符'''
    #print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))
    #交叉验证，找出出错的样本
    print(sum(model_selection.cross_val_score(clf, X, y, cv=5)))
    split_train, split_cv = model_selection.train_test_split(df, test_size = 0.3, random_state = 0) #这里的数据要包含passengerID便于匹配原始数据，查看数据的所有特征
    train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
    cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(cv_df.as_matrix()[:, 1:])
    origin_data_train = pd.read_csv("Train.csv")
    bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
    #print(bad_cases)

    '''通过learning curves观察模型是欠拟合还是过拟合'''
    plot_learning_curve(clf, u"Learning curve", X, y)  #图中看出cv误差与train误差相差不大，说明没有处于欠拟合状态，可以继续增加特征集或者选择更高项的多项式特征

    #考虑特征的改进
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()
    y = train_np[:, 0]
    X = train_np[:, 1:]
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    print(sum(model_selection.cross_val_score(clf, X, y, cv=5)))


    #然后做模型融合（model ensemble）,这里采用的是Bagging
    # bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,bootstrap_features=False, n_jobs=-1)
    # bagging_clf.fit(X, y)
    # predictions = bagging_clf.predict(test_df)
    # result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    # result.to_csv("logistic_regression_bagging_predictions.csv", index=False)





