#!/usr/bin/env python
# coding: utf-8

# ### House Price Prediction -  Kaggle
# 
# Models Used:
# 
# 1. Random Forest Regression
# 2. XGBoost Regression
# 
# 
# The goal of this project are: 
# 
# 1. to perform data cleaning and do comprehensive data analysis
# 2. make features normally distributed by removing Skewness and Kurtosis
# 2. to creat and test models which are well trained to predict sale prices of the houses
# 
# 

# In[2]:


# Importing necessary libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from scipy.stats import norm, skew
from scipy import stats
from scipy.special import boxcox1p


# In[4]:



#loading the the training dataset
train_df = pd.read_csv('train.csv')


# In[5]:


# Taking a view at our training dataset
train_df.head()


# In[6]:


# Figuring out how many features are categorical and numeric
train_df.get_dtype_counts()


# In[7]:


fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('GrLivArea', fontsize=11)
plt.show()


# In[8]:


train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)


# In[9]:


fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=11)
plt.xlabel('GrLivArea', fontsize=11)
plt.show()


# In[10]:


sns.distplot(train_df['SalePrice'], fit=norm)


# In[11]:


print("Skewness in SalePrice: %f" % train_df['SalePrice'].skew())
print("Kurtosis in SalePrice: %f" % train_df['SalePrice'].kurt())


# In[12]:


train_df["SalePrice"] = np.log1p(train_df["SalePrice"])


# In[13]:


sns.distplot(train_df['SalePrice'], fit=norm)


# In[14]:


print("Skewness in SalePrice: %f" % train_df['SalePrice'].skew())
print("Kurtosis in SalePrice: %f" % train_df['SalePrice'].kurt())


# In[15]:


# 1. Data Cleaning
# with the help of heatmap seeing which features have more missing values
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[16]:


# calculating the percentage of missing values, so we can think if that feature is worth to keep for analysis.
null_count = train_df.isnull().sum().sort_values(ascending = False)
null_percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)
missing_value = pd.concat([null_count, null_percent ], axis = 1)
missing_value.head(20)


# In[17]:


# droping those features which are having close to or more than 50% of data as missing values
train_df.drop(['Id', 'FireplaceQu', 'MiscFeature', 'Alley', 'Fence','FireplaceQu', 'PoolQC' ], axis = 1, inplace = True)


# In[18]:


train_df.isnull().sum().sort_values(ascending = False).sort_values(ascending = False).head(5)


# In[19]:


# since these features are categorical in nature but prersent in numeric values. Hence conveting then into string type.
train_df['MSSubClass'] = train_df['MSSubClass'].apply(str)
train_df['YrSold'] = train_df['YrSold'].astype(str)
train_df['MoSold'] = train_df['MoSold'].astype(str)


# In[20]:


# this function will replace the missing values for categorical feature with the most frequent category of that column, 
# and for numeric features with the mean of that column
def missing_value_impute(df):
    miss_sum = df.isnull().sum()
    miss_list = miss_sum[miss_sum > 0]
    
    for feature in list(miss_list.index):
        
        if df[feature].dtype == 'object':
            df[feature].fillna(df[feature].mode().index[0], inplace=True)
            
        elif df[feature].dtype != 'object':
            df[feature].fillna(df[feature].mean(), inplace=True)


# In[21]:


#applyin that function on our training data
missing_value_impute(train_df)


# In[22]:


# rechecking if we have left any missing value
sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[23]:


train_df.isnull().sum().sort_values(ascending = False).head(5)


# In[24]:


# checking the correlation among the features
corr=train_df.corr()
plt.figure(figsize= (20,20))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[75]:


#Correlation with output variable
cor_target = abs(corr["SalePrice"])
#Selecting highly correlated features
rel_feat = cor_target.sort_values()
rel_feat


# In[25]:


#sns.set()
#sns.pairplot(train_df, size = 2.5)
#plt.show()


# In[27]:


X_train=train_df.drop(['SalePrice'],axis=1)

y_train=train_df['SalePrice']

X_train.shape


# In[28]:


test_df= pd.read_csv('test_nonnorm.csv')
test_df.head(5)
test_df.shape


# In[29]:


final_df = pd.concat([X_train, test_df], axis = 0)
final_df.shape


# In[30]:


numeric_feats = final_df.dtypes[final_df.dtypes != "object"].index

skewed_feats = final_df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

print("\nSkewness in numerical features of dataset: \n")
skewness = pd.DataFrame({'Skewness' :skewed_feats})
skewness


# In[31]:


high_skew = skewed_feats[abs(skewed_feats) > 0.1]
skewed_features = high_skew.index
lam = 0.15
for feat in skewed_features:
    final_df[feat] = boxcox1p(final_df[feat], lam)


# In[32]:


final_df.head(5)


# In[33]:


final_df.shape


# In[ ]:





# In[30]:


# Here we are loading our cleaned test dataset.
# the reason we are loading test dataset is beacause after analysis we have figured out that the number of categories in some of
# the features are different in train and test data set.


# In[31]:


# There are 75 columns in train dataset, because there is target feature 'Saleprice' is present here.
#train_df.head(5)


# In[32]:


# Making a final dataset for feature engineering by joing test and train datasets.


# In[35]:


# Making a list of only categorical features for OneHotEncoding
df_object = final_df.dtypes == object
categorical_col = final_df.columns[df_object].tolist()
categorical_col


# In[36]:


new_df = final_df[['MSZoning', 'Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood', 'Condition2','BldgType','Condition1','HouseStyle','SaleType', 'SaleCondition','ExterCond', 'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC', 'CentralAir', 'Electrical','KitchenQual','Functional', 'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']]
new_df.head(5)


# In[37]:


# perofrming OneHotEncoding on categorical features of the dataset.
# To avoid the dummy variable trap, we are dropping the first dummy variable columns.
temp_df = pd.concat([pd.get_dummies(new_df[col], drop_first = True) for col in new_df], axis=1)
temp_df.shape


# In[38]:


# dropping the original columns for which we have created dummy variables
for col in categorical_col:
    final_df.drop([col], axis = 1, inplace= True)


# In[39]:


final_df.shape


# In[40]:


# finally concatinating our dummy variable with our original dataset
onehot_df = pd.concat([final_df, temp_df], axis = 1)
onehot_df.shape


# In[41]:


onehot_df.head(5)


# In[42]:


# Dropping all the duplicate columns
onehot_df =onehot_df.loc[:,~onehot_df.columns.duplicated()]
onehot_df.shape


# In[43]:


# since we have created dummy variable for every category of every feau=ture, lets seperate back our train and test dataset.
X_train = onehot_df.iloc[:1458, :]
test_set = onehot_df.iloc[1458:, :]


# In[44]:


#test_set = test_set.drop(['SalePrice'], axis =1)
test_set.shape


# In[45]:


X_train.shape


# In[44]:


# Seperating out target and predictor variables for model training
#X_train=train_set.drop(['SalePrice'],axis=1)
#y_train=train_set['SalePrice']


# In[46]:


# Model Creation
# Random Forest Regression will be the first model to predict hous sale price.

RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)


# In[47]:


# Since we are going to perform the hyperparameter tuning for this model.
# Hence creating a matix of parameters to train our model.

bootstrap=  [True, False]
max_depth= [2, 3, 5, 10, None]
max_features = ['auto', 'sqrt']
min_samples_leaf= [1, 2, 4]
min_samples_split= [2, 5, 10]
n_estimators= [50, 100, 200, 300, 500, 700]


# In[48]:


forest_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[49]:


# Using the RandomizedSearchCV for hyper-parameter tuning
RF_rdcv = RandomizedSearchCV(estimator=RF_model,
            param_distributions=forest_grid,
            cv=5, n_iter=10,
            scoring = 'neg_mean_absolute_error', verbose = 3, n_jobs = -1,
            return_train_score = True,
            random_state=42)


# In[50]:


RF_rdcv.fit(X_train, y_train)


# In[51]:


# With the help of best estimator we can have a set of parametes which give the best score.
RF_rdcv.best_estimator_


# In[52]:


# Retraining our model on the best parameters possible
RF_model = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,
                      max_features='sqrt', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=2, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=50,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
RF_model.fit(X_train, y_train)


# In[53]:


# Predicting the sale price of houses in the test dataset.
RF_pred = RF_model.predict(test_set)
RF_df = pd.DataFrame(RF_pred, columns = ['RF_pred'])
RF_df.head()


# In[54]:


RF_pred_nat = np.floor(np.expm1(RF_model.predict(test_set)))
RF_pred_nat = pd.DataFrame(RF_pred_nat, columns = ['RF_pred'])
RF_pred_nat.head()


# In[55]:


# cheacking the score of out model performance
RF_model.score(X_train, y_train)


# In[56]:


# XGBoost regression will be our second model to predict the sale prices of the houses

XGB_model=xgboost.XGBRegressor(objective ='reg:squarederror')
XGB_model.fit(X_train, y_train)


# In[57]:


# Creating a matix of parameters for the hyper parameter tunning of this model
n_estimators = [50, 100, 200, 300, 500]
max_depth = [2, 3, 5, 10]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]


# In[58]:


XGB_hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[59]:


# Using the RandomizedSearchCV for hyperparameter tuning
XGB_rdcv = RandomizedSearchCV(estimator=XGB_model,
            param_distributions=XGB_hyperparameter_grid,
            cv=5, n_iter=10,
            scoring = 'neg_mean_absolute_error', verbose = 3, n_jobs = -1,
            return_train_score = True,
            random_state=42)


# In[60]:



XGB_rdcv.fit(X_train,y_train)


# In[61]:


XGB_rdcv.best_estimator_


# In[62]:


XGB_model=xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.2, max_delta_step=0,
             max_depth=2, min_child_weight=2, missing=None, n_estimators=500,
             n_jobs=1, nthread=None, objective='reg:squarederror',
             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
             seed=None, silent=None, subsample=1, verbosity=1)


# In[63]:


# Retraining our model on the best parameters possible
XGB_model.fit(X_train, y_train)


# In[64]:


# Predicting the sale price of houses in the test dataset with XGBoost.
XGB_pred = XGB_model.predict(test_set)
XGB_pred = np.floor(np.expm1(XGB_pred))
XGB_df = pd.DataFrame(XGB_pred, columns = ['SalePrice'])
XGB_df


# In[65]:


# Checking our model performance score
XGB_model.score(X_train, y_train)


# In[66]:


# Just out of curiosity, checking how close is our prediction with the given sale price in train set.
#train_pred = XGB_model.predict(X_train)

train_pred = np.expm1(XGB_model.predict(X_train))

train_pred = pd.DataFrame(train_pred)
pred_compr = pd.concat([train_pred, y_train ], axis =1)
pred_compr.columns = ['pred_saleprice', 'given_saleprice']
pred_compr.head()


# In[67]:


# for the sake of submission on kaggle , creating a seperate submission file for Kaggle competition.
test_data= pd.read_csv('test.csv')


# In[68]:


test_id = test_data.iloc[:, :1]
test_id.shape


# In[69]:


XGB_df


# In[71]:


XGB_df
output_file = pd.concat([test_id, XGB_df], axis =1)


# In[72]:


output_file.to_csv('trntst_submission_file.csv', index = False)

