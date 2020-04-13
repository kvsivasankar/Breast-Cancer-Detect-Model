
# coding: utf-8

# <div class="alert alert-info">
# ** Common helper functions **
# 
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import missingno as mn
color = sns.color_palette()
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# In[6]:


class CommonHelper:
    def __init__(self, df):
        self.df = df
    
    def missing_info(self):
        for i,j in sorted(zip(pd.isnull(self.df).sum().index,round(100*pd.isnull(self.df).sum()/self.df.shape[0],1)),key = lambda x: -x[1]):
            if j > 0:
                print(i,"---"*3,self.df[i].dtypes,"---"*4,j,"%")    
            
    def missing_info2(self):
        df_na = (self.df.isnull().sum() / len(df)) * 100
        df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :df_na})
        print(missing_data)
       
    def get_column_count(self):
        int_columns = [col for col in self.df.columns if(df[col].dtype != "object")]
        print("No of integer type columns:",len(int_columns))
        print(int_columns)
        print("")
        obj_columns = [col for col in self.df.columns if(df[col].dtype == "object")]
        print("No of object type columns:",len(obj_columns))
        print(obj_columns)
        return int_columns,obj_columns

    def describe_data(self):
        print("checking missing data information: \n")
        
        print(self.df.isnull().any())
        print('\n')
        print("---"*6)
        print('\n')
        print("number of observations in dataset: {}".format(self.df.shape))
        print('\n')
        print("---"*6)
        print('\n')
        print("number of features in dataset: \n")
        print(self.df.columns.values)
        print('\n')
        print("---"*6)
        print('\n')
        print("dataset information : \n")
        print(self.df.info())
        print('\n')
        print("---"*6)
        print('\n')
        print("describing data: \n")
        print(self.df.describe())
        
    
    def separate_numaric_categorical_features(self):
        #separate variables into new data frames
        numeric_data = self.df.select_dtypes(include=[np.number])
        cat_data = self.df.select_dtypes(exclude=[np.number])
        print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))
        return numeric_data,cat_data
       
    def fill_na_value(self,column_names,replace_with):
        for col in column_names:
            self.df[col] = self.df[col].fillna(replace_with)
            return self.df

    def fill_na_with_zero(self,column_names):
        for col in column_names:
            self.df[col] = self.df[col].fillna(0)
            return self.df
            
    def convert_numarict_to_categorical(self,column_names=[]):
        self.df[column_names] = self.df[column_names].apply(str)
        
    def drop_columns(self,column_names=[]):
        self.df = self.df.drop(column_names, axis=1)
        return self.df
    
    def target_mapping(self,target_col='',map_dic={}):
        self.df['Target'] = self.df[target_col].map(map_dic)
        return self.df
    
    def target_distribution(self,target_col=''):
        print(self.df[target_col].value_counts())
        print('\n')
        print(self.df[target_col].value_counts(normalize=True))
        
    def show_countplots(self,column_names=[]):
        for col in column_names:
            if(len(self.df[col].value_counts()) < 5):
                plt.figure(figsize=(5,5))
            else:
                plt.figure(figsize=(12,6))
            print(sns.countplot(x=col, data=self.df, palette="muted"))
            plt.show()
        
