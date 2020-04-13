
# coding: utf-8

# 
# <div class="alert alert-info">
# ** EDA helper functions **
# 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler


# In[3]:


class EDAHelper:
    def __init__(self,df):
        self.df = df
        
    def check_correlation(self,df):
        plt.subplots(figsize = (15,8))
        sns.heatmap(df.corr(), annot=True,cmap="Blues")
        
    def logtransformation(self,df,col_names=[]):
        log_means = np.log1p(df[col_names])
        skewness = pd.DataFrame({'Original Skewness':df[col_names].skew(),'Log Transformed':log_means.skew()})
        skewness['Skewness Reduction'] = skewness['Original Skewness'] - skewness['Log Transformed']
        print(skewness)
        
    def check_skewness_numaric_columns(self,df):
        numeric_feats = [col for col in df.columns if df[col].dtypes != 'object']
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        print(skewness)
        
    def check_skewness(self,df,col_names=[]):
        print(df[col_names].skew())
                                 

    def boxcox_transformation(self,df):
        numeric_feats = [col for col in df.columns if df[col].dtypes != 'object']
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})

        skewed_features = skewness.index
        bx_cx_criteria = 0.15   #### should be b/w 0 and 1 for our case
        for feat in skewed_features:
            df[feat] = boxcox1p(df[feat], bx_cx_criteria)
        return df

    def label_encoding(self,df,col_names=[]):
        for col in col_names:
            lbl = LabelEncoder()
            lbl.fit(list(df[col].values)) 
            df[c] = lbl.transform(list(df[c].values))
        # shape        
        print('Shape all_data: {}'.format(df.shape))
        return df

    def standardscaling(self,df):
        numeric_features = [f for f in df.columns if df.dtypes[f] != 'object']

        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        scaled = scaler.transform(df[numeric_features])
        for i, col in enumerate(numeric_features):
            df[col] = scaled[:,i]
        return df
        
    def create_dummies(self,df):
        self.df = pd.get_dummies(self.df,drop_first=True)
        print(df.shape)
        return df

    def categorical_summarized(self, df,x=[], y=[], hue=[], palette='Set1', verbose=True):
        '''
        Helper function that gives a quick summary of a given column of categorical data
        Arguments
        =========
        dataframe: pandas dataframe
        x: str. horizontal axis to plot the labels of categorical data, y would be the count
        y: str. vertical axis to plot the labels of categorical data, x would be the count
        hue: str. if you want to compare it another variable (usually the target variable)
        palette: array-like. Colour of the plot
        Returns
        =======
        Quick Stats of the data and also the count plot
        '''
        if x == None:
            column_interested = y
        else:
            column_interested = x
        series = df[column_interested]
        print(series.describe())
        print('mode: ', series.mode())
        if verbose:
            print('='*80)
            print(series.value_counts())

        sns.countplot(x=x, y=y, hue=hue, data=df, palette=palette)
        plt.show()
        
    def quantitative_summarized(self,df ,x=[], y=[], hue=[], palette='Set1', ax=None, verbose=True, swarm=False):
        '''
        Helper function that gives a quick summary of quantattive data
        Arguments
        =========
        dataframe: pandas dataframe
        x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
        y: str. vertical axis to plot the quantitative data
        hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
        palette: array-like. Colour of the plot
        swarm: if swarm is set to True, a swarm plot would be overlayed
        Returns
        =======
        Quick Stats of the data and also the box plot of the distribution
        '''
        series = df[y]
        print(series.describe())
        print('mode: ', series.mode())
        if verbose:
            print('='*80)
            print(series.value_counts())

        sns.boxplot(x=df[x], y=df[y], hue=hue, data=df, palette=palette, ax=ax)

        if swarm:
            sns.swarmplot(x=x, y=y, hue=hue, data=df,
                          palette=palette, ax=ax)

        plt.show()
        
    def show_boxplot(self,df,x=[], y=[]):
        sns.boxplot(x=df[x],y=df[y])
        x = plt.xticks(rotation=90)
        
    def show_histogram(self,df,col_names=[]):
        df[col_names].hist(figsize=(10,12), bins=20, layout=(5,2), grid=False)
        plt.tight_layout();

    def show_pairplots(self,df,col_names=[],hue_col=None):
        p = sns.pairplot(df[col_names], hue=hue_col,plot_kws={'alpha':0.6}, palette='magma')
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        handles = p._legend_data.values()
        labels = p._legend_data.keys()
        p.fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2)
        p.fig.set_dpi(80);

    def show_heatmap(self,df):
        fig, ax = plt.subplots(figsize=(10,8)) 
        ax = sns.heatmap(df.corr(), cmap="YlGnBu",annot=True, linewidths=.5, ax=ax)
# In[4]:


# this applies only for numaric columns
class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print('Dropping {} with vif {}'.format(X.columns[maxloc],max_vif))
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X


# In[ ]:


#transformer = ReduceVIF()
#all_data_VIF_excl = transformer.fit_transform(all_data[all_data.columns], y=None)

#all_data_VIF_excl.head()

