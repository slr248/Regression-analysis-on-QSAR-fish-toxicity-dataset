# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:40:36 2019

@author: sanil
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
import pickle
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LinearRegression, Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

names=['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP','quantitative response, LC50 [-LOG(mol/L)]']
data=pd.read_csv('C:\\Users\\sanil\\Desktop\\EE660\\project\\project_files\\data\\qsar_fish_toxicity.csv',delimiter=';',names=names,na_values=["n/a","na"])
y=data[names[-1]]
X=data.drop(columns=names[-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

#BASE LINE MODELS
class Base_Model(BaseEstimator,RegressorMixin): #baseEstimator is the base class for all estimators, imported from sklearn.base
    def __init__(self,models):
        self.models=[clone(i) for i in models] #clones the models and the params passed to models
    def fit(self,X,y):
        for model in self.models:
            model.fit(X,y)
        return self
    def predict(self,X):
        predictions=[]
        for model in self.models:
            predictions.append(model.predict(X))
        return predictions
    def score(self,X,y):
        model_score=[]
        for model in self.models:
            model_score.append(model.score(X,y))
        return model_score
    def rmse(self,X,y):
        rmse_base=[]
        predictions=[]
        for model in self.models:
            predictions.append(model.predict(X))
        for p in predictions:
            rmse_base.append(np.sqrt(mean_squared_error(y,p)))
        return rmse_base
        
    
    
#########DATASET###############################################################################
print('DATASET:')
print('Total number of samples:{0}'.format(len(data)))
print('number of training samples:{0}'.format(len(X_train)))
print('number of testing samples:{0}'.format(len(X_test)))
print('number of features:{0}'.format(len(names)-1))


###############################################################################################
lasso=Lasso(alpha=0.0005,random_state=1)
LR=LinearRegression()
ridge=Ridge(alpha=1.0)
Enet=ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
rf=RandomForestRegressor(n_estimators=3,bootstrap=True)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
ada=AdaBoostRegressor(n_estimators=1000)

base_models = pickle.load(open('base_models.sav', 'rb'))
###########Training error######################################################################
print('')
Ein_base_models=base_models.rmse(X_train,y_train)
print('BASELINE MODELS TRAINING ERROR (RMSE)')
base_model_names=['lasso','Linear Regression','Ridge Regression','Elasticnet','random forest','Gradient Boosting','AdaBoost']
for i in range(len(Ein_base_models)):
    print(base_model_names[i]+': '+str(Ein_base_models[i]))


###########Test set############################################################################
print(' ')
Eout_base_models=base_models.rmse(X_test,y_test)
print('BASELINE MODELS TEST SET ERROR (RMSE)')
base_model_names=['lasso','Linear Regression','Ridge Regression','Elasticnet','random forest','Gradient Boosting','AdaBoost']
for i in range(len(Eout_base_models)):
    print(base_model_names[i]+': '+str(Eout_base_models[i]))

############Tuned models######################################################################
    
best_alphas={'ridge': 3, 'lasso': 0.001, 'Enet': 0.001}

lasso=Lasso(alpha=best_alphas['lasso'],random_state=1)
LR=LinearRegression()
ridge=Ridge(alpha=best_alphas['ridge'])
Enet=ElasticNet(alpha=best_alphas['Enet'], l1_ratio=.9, random_state=3)
rf=RandomForestRegressor(n_estimators=50,max_depth=6,bootstrap=True)
ada=AdaBoostRegressor(n_estimators=3000,learning_rate=1)
GB=GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01)

tuned_models=pickle.load(open('tuned_base.sav', 'rb'))

#TRAINING ERROR
print(' ')
Ein_tuned_models=tuned_models.rmse(X_train,y_train)
print('TUNED BASELINE MODELS TRAINING ERROR (RMSE)')
base_model_names=['lasso','Linear Regression','Ridge Regression','Elasticnet','random forest','Gradient Boosting','AdaBoost']
for i in range(len(Ein_tuned_models)):
    print(base_model_names[i]+': '+str(Ein_tuned_models[i]))

#TESTING ERROR
print(' ')
Eout_tuned_models=tuned_models.rmse(X_test,y_test)
print('TUNED BASELINE MODELS TEST SET ERROR (RMSE)')
base_model_names=['lasso','Linear Regression','Ridge Regression','Elasticnet','random forest','Gradient Boosting','AdaBoost']
for i in range(len(Eout_tuned_models)):
    print(base_model_names[i]+': '+str(Eout_tuned_models[i]))
    
###########################STACKING###########################################################

class Stacked_models(BaseEstimator,RegressorMixin,TransformerMixin): #thanks to https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    def __init__(self,base_models,meta_model,n_folds):
        self.base_models=[clone(i) for i in base_models] 
        self.meta_model=clone(meta_model)
        self.n_folds=n_folds
    
    def fit(self,X,y):
        X=np.array(X)
        y=np.array(y)
        kf=KFold(n_splits=self.n_folds,shuffle=True)
        
        new_feat=np.zeros((X.shape[0],len(self.base_models))) #new_feat is (num_rows of training data x num base models), it contains the predictions of each base model on the training data
        for i in range(0,len(self.base_models)):
            for train_index,test_index in kf.split(X,y):
                model=self.base_models[i]
                model.fit(X[train_index],y[train_index]) #fit current base model to the current training fold
                pred=model.predict(X[test_index]) #get predictions on the kth fold
                new_feat[test_index,i]=pred         #save predictions at the kth fold indexes of the training data in new-feat
        self.meta_model.fit(new_feat,y)
        return self
    
    def get_feat(self,X,y):
        X=np.array(X)
        y=np.array(y)
        kf=KFold(n_splits=self.n_folds,shuffle=True)
        
        new_feat=np.zeros((X.shape[0],len(self.base_models))) #new_feat is (num_rows of training data x num base models), it contains the predictions of each base model on the training data
        for i in range(0,len(self.base_models)):
            for train_index,test_index in kf.split(X,y):
                model=self.base_models[i]
                model.fit(X[train_index],y[train_index]) #fit current base model to the current training fold
                pred=model.predict(X[test_index]) #get predictions on the kth fold
                new_feat[test_index,i]=pred         #save predictions at the kth fold indexes of the training data in new-feat
        
        return new_feat
    def predict(self,X):
        test_feat=np.zeros((X.shape[0],len(self.base_models)))
        for i,model in enumerate(self.base_models):
            test_feat[:,i]=model.predict(X)
        pred=self.meta_model.predict(test_feat)
        return pred


stacked_models=['ElasticNet','Gradient Boosting','AdaBoost','RandomForest']#Enet,GB,ada,rf
meta_model='lasso'
print(' ')
print('STACKED MODELS:')
for i in stacked_models:
    print(i)

print('')
print('meta model: '+meta_model)

stacked=pickle.load(open('stacked_model.sav', 'rb'))
#TRAINING
pred_train=stacked.predict(X_train)
train_error=np.sqrt(mean_squared_error(y_train,pred_train))
print('')
print('training error: '+str(train_error))

#TESTING
pred=stacked.predict(X_test)
test_error=np.sqrt(mean_squared_error(y_test,pred))
print('')
print('test error: '+str(test_error))
