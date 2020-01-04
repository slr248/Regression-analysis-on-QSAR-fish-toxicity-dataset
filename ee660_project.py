#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
import pickle


# In[3]:


names=['CIC0','SM1_Dz(Z)','GATS1i','NdsCH','NdssC','MLOGP','quantitative response, LC50 [-LOG(mol/L)]']
data=pd.read_csv('C:\\Users\\sanil\\Desktop\\EE660\\project\\project_files\\qsar_fish_toxicity.csv',delimiter=';',names=names,na_values=["n/a","na"])
y=data[names[-1]]
X=data.drop(columns=names[-1])


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)


# In[6]:


data.head()


# In[12]:


len(data)


# In[136]:


"""for i in names:
    if data[i].isnull().values.any():
        print('yes')
    else:
        print('no missing values')"""


'''data preprocessing
1.standardize and normalize
2.feature engineering, find relationships b/w features
3. Choose a classificatin dataset as well?
4.plot gaussians'''

fig, ax = plt.subplots()
ax.scatter(x = data['NdsCH'], y = data['NdssC'])
plt.ylabel('CIC0', fontsize=13)
plt.xlabel('SM1_Dz', fontsize=13)
plt.show()


# # Q-Q plot and plot of Quantitative response fit with a normal distribution

# In[93]:


#gaussians
#mu,sigma=norm.fit(ytrain)
sns.distplot(y,fit=norm)
plt.show()
res = stats.probplot(y, plot=plt)


# # Distribution of features

# In[138]:


def plot_feature_dist(data,columns,*args):
    fig,axes=plt.subplots(ncols=len(columns),figsize=(30,5))
    for axes,col in zip(axes,columns):
        sns.distplot(data[col],ax=axes)
        plt.suptitle('feature distribution')
plot_feature_dist(X,X.columns)


# In[140]:


def feature_vs_target(data,ytrain):
    fig,axes=plt.subplots(figsize=(20,5))#nrows=2,ncols=3,figsize=(15,5))
    fig.tight_layout()
    '''for axes,cols in zip(axes,data.columns):
        axes=plt.scatter(data[cols],ytrain)
        plt.xlabel(cols)
        plt.ylabel('Quantitative response')
        plt.show()'''
    col_names=X_train.columns
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.scatter(X_train[col_names[i]],y_train)
        plt.xlabel(col_names[i])
        plt.ylabel('quantitative response')


feature_vs_target(X,y)


# In[144]:


#heat map
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9,annot=corrmat,square=True)


# #linear regression

# # BASE MODELS

# In[198]:


#scoring metric :ROOT mean squared error
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

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



# In[199]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LinearRegression, Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


# In[200]:


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


# In[202]:


base_models=Base_Model(models=(lasso,LR,ridge,Enet,rf,GBoost,ada))
base_models.fit(X_train,y_train)
#base_models.score(X_train,y_train)
#base_models.predict(X_train)
#base_models.rmse(X_train,y_train) #RMSE on training for the base Models
#base_models.rmse(X_test,y_test)  #RMSE on test set for base models
filename = 'base_models.sav'
pickle.dump(base_models, open('C:\\Users\\sanil\\Desktop\\EE660\\project\\project_files\\'+str(filename), 'wb'))
# # Regularized linear models CrossValidation and hyper parameter tuning
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.rmse(X_test,y_test)

# In[187]:


#RMSE calculator
def rmse(model):
    #5 fold CV on the training set where the -ve mean squared error is calculated for each fold, so we take sqrt of the -ve of that value
    rmse=np.sqrt(-cross_val_score(model,X_train,y_train,scoring="neg_mean_squared_error",cv=5))
    return np.mean(rmse)


# In[188]:


alphas=[0.001,0.005,0.01,0.05,0.1,0.3,1,3,5,10,15,30,50,70] #degree of regularization
cv_ridge=[rmse(Ridge(alpha=a)) for a in alphas]
cv_lasso=[rmse(Lasso(alpha=a)) for a in alphas]
cv_Enet=[rmse(ElasticNet(alpha=a)) for a in alphas]


# In[189]:


plt.plot(alphas,cv_ridge)
plt.plot(alphas,cv_lasso)
plt.plot(alphas,cv_Enet)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.legend(['Ridge','Lasso','ElasticNet'])
plt.show()


# In[190]:


best_alphas={'ridge':alphas[cv_ridge.index(min(cv_ridge))],'lasso':alphas[cv_lasso.index(min(cv_lasso))],'Enet':alphas[cv_Enet.index(min(cv_Enet))]}


# In[191]:


for key, value in best_alphas.items():
    print(key, value)


# In[204]:


lasso=Lasso(alpha=best_alphas['lasso'],random_state=1)
#KR=KernelRidge(alpha=best_alphas['kernel'],kernel='polynomial',degree=2)
ridge=Ridge(alpha=best_alphas['ridge'])
Enet=ElasticNet(alpha=best_alphas['Enet'], l1_ratio=.9, random_state=3)


# In[205]:


reg_lin_models=Base_Model(models=(lasso,ridge,Enet))
reg_lin_models.fit(X_train,y_train)
#base_models.score(X_train,y_train)
#base_models.predict(X_train)
#print(reg_lin_models.rmse(X_train,y_train)) #RMSE on training for the base Models
print(reg_lin_models.rmse(X_test,y_test))


# # A study on overfitting

# In[106]:


Lr=LinearRegression()
Lr.fit(X_train,y_train)
Lr.coef_


# In[107]:


from sklearn.decomposition import PCA
pca=PCA(n_components=1)
X_train_pca=pca.fit_transform(X_train)


# In[108]:


plt.scatter(X_train_pca,y_train)
plt.xlabel('training data after PCA')
plt.ylabel('Quantitative response')
plt.show()


# In[ ]:





# # Regression tree parameter tuning

# In[33]:


from sklearn.model_selection import GridSearchCV
#RANDOM FOREST REGRESSOR
grid=GridSearchCV(estimator=RandomForestRegressor(),param_grid={'n_estimators':[10,50,100,1000],'max_depth':[i for i in range(3,7)]},cv=5,scoring='neg_mean_squared_error')


# In[32]:


grid_rf=grid.fit(X_train,y_train)


# In[39]:


grid_rf.best_params_


# In[44]:


rf=RandomForestRegressor(n_estimators=50,max_depth=6,bootstrap=True)
rf.fit(X_train,y_train)
mean_squared_error(y_test,rf.predict(X_test))


# In[39]:


#ADABOOST REGRESSOR
params_ada={'n_estimators':[10,50,100,1000,3000],'learning_rate':[0.01,0.05,0.3,0.1,1]}
grid_ada=GridSearchCV(estimator=AdaBoostRegressor(loss='exponential'),param_grid=params_ada,cv=5,scoring='neg_mean_squared_error')


# In[40]:


grid_ada_=grid_ada.fit(X_train,y_train)


# In[41]:


grid_ada_.best_params_


# In[42]:


ada=AdaBoostRegressor(n_estimators=1000,learning_rate=1)
ada.fit(X_train,y_train)
mean_squared_error(y_test,ada.predict(X_test))


# In[43]:


#GRADIENT BOOSTING REGRESSOR
grid_gb=GridSearchCV(estimator=GradientBoostingRegressor(loss='huber'),param_grid=params_ada,cv=5,scoring='neg_mean_squared_error')


# In[44]:


grid_gb_=grid_gb.fit(X_train,y_train)


# In[46]:


grid_gb_.best_params_


# In[47]:


GB=GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01)
GB.fit(X_train,y_train)
mean_squared_error(y_test,GB.predict(X_test))


# # Stacking Models for better predictions

# In[195]:


class Stacked_models(BaseEstimator,RegressorMixin,TransformerMixin): ##thanks to https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
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



# In[181]:


lasso=Lasso(alpha=best_alphas['lasso'],random_state=1)
#KR=KernelRidge(alpha=best_alphas['kernel'],kernel='polynomial',degree=2)
LR=LinearRegression()
ridge=Ridge(alpha=best_alphas['ridge'])
Enet=ElasticNet(alpha=best_alphas['Enet'], l1_ratio=.9, random_state=3)
rf=RandomForestRegressor(n_estimators=50,max_depth=6,bootstrap=True)
ada=AdaBoostRegressor(n_estimators=3000,learning_rate=1)
GB=GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01)


# In[232]:
tuned_base=Base_Model(models=(lasso,LR,ridge,Enet,rf,GB,ada))
tuned_base.fit(X_train,y_train)
pickle.dump(tuned_base, open('C:\\Users\\sanil\\Desktop\\EE660\\project\\project_files\\tuned_base.sav', 'wb'))

stacked=Stacked_models(base_models=(Enet,GB,ada,rf),meta_model=lasso,n_folds=5)
stacked.fit(X_train,y_train)
pickle.dump(stacked, open('C:\\Users\\sanil\\Desktop\\EE660\\project\\project_files\\stacked_model.sav', 'wb'))


# In[234]:


meta_feat=stacked.get_feat(X_train,y_train)


# In[236]:


meta_feat_=pd.DataFrame(meta_feat,columns=['ElasticNet','Gradient Boosting','AdaBoost','random forest regressor'])


# In[237]:


meta_feat_.head()


# In[120]:


X_train.head()


# In[233]:


pred=stacked.predict(X_test)
np.sqrt(mean_squared_error(y_test,pred))


# In[6]:


lasso.fit(X_train,y_train)
np.sqrt(mean_squared_error(y_test,lasso.predict(X_test)))


# # Neural network as our meta model
