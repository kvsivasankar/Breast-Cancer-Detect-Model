
# coding: utf-8

# ### About Tumors
# 
# Tumors are abnormal growths in your body. They are made up of extra cells. Normally, cells grow and divide to form new cells as your body needs them. When cells grow old, they die, and new cells take their place. Sometimes, this process goes wrong. New cells form when your body does not need them, and old cells do not die when they should. When these extra cells form a mass, it is called a tumor. 
# 
# Tumors can be either **benign** or **malignant**. Benign tumors aren't cancer. Malignant ones are. Benign tumors grow only in one place. They cannot spread or invade other parts of your body. Even so, they can be dangerous if they press on vital organs, such as your brain.
# 
# ## key differences between benign and malignant tumors?
# <img src="benign_malignant.png" />
# 
# ### Problem solving approach
# 
# This dataset contains information on 569 breast tumors and the mean, standard error and worst measures for 10 different properties. I start with an EDA analysing each properties' distribution, followed by the pair interactions and then the correlations with our target.
# 
# After the EDA I set up 8 out-of-the-box models for a first evaluation and use stratified cross-validation to measure them. I use **Recall** instead of **Accuracy or F1-Score** since I want to detect all malignant tumors. 
# 
# After the first results I analyse features importances, do a single round of feature selection and evaluate the models again. At the end I analyse model errors from the 8 first models I choosen 5 models for fine tuning: 
# **Logistic Regression, SVC, Random Forest, Gradient Boosting and KNN.**
# 
# Then I proceed to tune the five models using **GridSearchCV** and prepare the data for model stacking by predicting probabilities for both train and test sets. 
# 
# Finally, I test all first level models and the stacked Logistic Regression on our untouched test-set. For the first level models, using regular 0.5 threshold Logistic Regression performed best with 95,8% Recall. By lowering the threshold SVC and Logistic Regression tied with over 98% recall with SVC having a higher Accuracy. By using the model-stacking technique, Logistic Regression was able to obtain 100% Recall on the test set. On the last chapter I summarize the findings and conclusions.
# 
# On Annex - A I repeat a few Machine Learning steps using SMOTE to generate new data points making the data balanced. 
# 
# On Annex - B I use three different dimensionality reduction techniques to see if I can reduce the dataset and still get a good test score.

# # Dataset
# 

# ### General Information
# 
# - Original format: csv
# - Dataset shape: 569 x 33 (rows x columns)
# - There are no null values in this data.
# - The values are in different scales
# 
# ### Features information
# 
# For each sample ten properties were measured:
# 
# <ol>
#     <li><b>Radius</b> - Mean distances from center to points on the perimeter</li>
#     <li><b>Texture</b> - Standard deviation of gray scale values</li>
#     <li><b>Perimeter</b></li>
#     <li><b>Area</b></li>
#     <li><b>Smoothness</b> - Local variation in radius lengths</li>
#     <li><b>Compactness</b> - Perimeter^2/Area - 1</li>
#     <li><b>Concavity</b> - Severity of concave portions of the contour</li>
#     <li><b>Concave points</b> - Number of concave portions of the contour</li>
#     <li><b>Simmetry</b></li>
#     <li><b>Fractal Dimension</b> - Coastline approximation - 1 </li>
# </ol>
# 
# 
# And for each of these properties we have three calculated values:
# - **Mean**
# - **Standard Error**
# - **Worst** (Average of the 3 largest values)
# 
# All the measures are float types.
# 
# ### Target
# 
# Our target is the categorical column either B (benign) or M (malignant).<br>
# There are 357 benign classes and 212 malignant classes - roughly **37% malignant tumors**.

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_columns', 40)

bc_df = pd.read_csv('data.csv')
bc_df.sample(10)


# ### Importing CommonHelper class
# 

# In[3]:


import os, importlib, sys

def callfunc(file_name, func_name, *args):
    pathname, filename = os.path.split(file_name)
    sys.path.append(os.path.abspath(pathname))
    modname = os.path.splitext(filename)[0]
    module = importlib.import_module(modname)
    result = getattr(module, func_name)(*args)
    return result

common_functions = callfunc(os.getcwd()+"/CommonHelper.py", "CommonHelper", bc_df)


# In[4]:


common_functions.describe_data()


# In[5]:


common_functions.missing_info()


# Removed 'id' and 'Unnamed: 32' columns from dataframe 

# In[6]:


bc_df = common_functions.drop_columns(['id','Unnamed: 32'])


# In[7]:


bc_df.tail()


# In[8]:


bc_df = common_functions.target_mapping(target_col='diagnosis',map_dic={'B':0, 'M':1})


# In[9]:


# For visualization purpose gathered same group of features
mean_feats = np.concatenate([['diagnosis'], bc_df.iloc[:,1:11].columns.tolist()])
error_feats = np.concatenate([['diagnosis'], bc_df.iloc[:,11:21].columns.tolist()])
worst_feats = np.concatenate([['diagnosis'], bc_df.iloc[:,21:31].columns.tolist()])


# In[10]:


mean_feats, error_feats, worst_feats


# In[11]:


# Target distribution in data
common_functions.target_distribution(target_col='diagnosis')


# In[12]:


common_functions.show_countplots(column_names=['diagnosis'])


# ## Split data into Train/Test sets

# In[13]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(bc_df, test_size=0.3,stratify=bc_df['Target'], random_state=123)


# In[14]:


train_count = train.diagnosis.value_counts(normalize=True)
test_count = test.diagnosis.value_counts(normalize=True)
pd.concat([train_count,test_count],axis=1)


# ## EDA

# In[15]:


# Importing EDA helper functions class
eda_helper = callfunc(os.getcwd()+"/EDAHelper.py", "EDAHelper",bc_df)


# In[16]:


eda_helper.check_skewness(train,mean_feats)
eda_helper.show_histogram(train,mean_feats)


# We can see that some features are pretty skewed. We can measure its skewness using pandas skew method and we can try comparing it to a log transformation of the same values to see if we can reduce the skewness.

# In[17]:


log_means = np.log1p(train.iloc[:,1:11])

skewness = pd.DataFrame({'Original Skewness':train.iloc[:,1:11].skew(),
                         'Log Transformed':log_means.skew()})
skewness['Skewness Reduction'] = skewness['Original Skewness'] - skewness['Log Transformed']
skewness


# In[18]:


eda_helper.check_skewness(train,error_feats)
eda_helper.show_histogram(train,error_feats)


# In[19]:


log_means = np.log1p(train.iloc[:,11:21])

skewness = pd.DataFrame({'Original Skewness':train.iloc[:,11:21].skew(),
                         'Log Transformed':log_means.skew()})
skewness['Skewness Reduction'] = skewness['Original Skewness'] - skewness['Log Transformed']
skewness


# In[20]:


eda_helper.check_skewness(train,worst_feats)
eda_helper.show_histogram(train,worst_feats)


# In[21]:


log_means = np.log1p(train.iloc[:,21:31])

skewness = pd.DataFrame({'Original Skewness':train.iloc[:,21:31].skew(),
                         'Log Transformed':log_means.skew()})
skewness['Skewness Reduction'] = skewness['Original Skewness'] - skewness['Log Transformed']
skewness


# ## Checking correlation
# 
# - We can use seaborn's amazing pairplot to give a first overview on all features and some pair interactions.
# - By using heatmap we can see feature correlation. But now I am not removing any features based on correlation. We will decide later part

# In[22]:


eda_helper.show_pairplots(train,mean_feats,hue_col='diagnosis')
eda_helper.show_heatmap(train[mean_feats])
#mean_feats


# In[23]:


eda_helper.show_pairplots(train,error_feats,hue_col='diagnosis')
eda_helper.show_heatmap(train[error_feats])


# In[24]:


eda_helper.show_pairplots(train,worst_feats,hue_col='diagnosis')
eda_helper.show_heatmap(train[worst_feats])


# ## Correlaton on Target

# In[25]:


fig, ax = plt.subplots(figsize=(20,15)) 
ax = sns.heatmap(train.corr(), cmap="YlGnBu",annot=True, linewidths=.5, ax=ax)


# In[26]:


corrs = train.corr()[["Target"]].sort_values("Target",ascending=False)[1:].reset_index()
corrs


# In[27]:


def feat_class(feat):
    if 'worst' in feat:
        return 'Worst'
    elif 'mean' in feat:
        return 'Mean'
    elif 'se' in feat:
        return 'Standard Error'


# In[28]:


corrs.rename(columns={'index':'Features'}, inplace=True)
corrs['Class'] = corrs['Features'].apply(feat_class)


# In[29]:


fig, ax = plt.subplots(figsize=(8,7), dpi=80)
sns.barplot(data=corrs, x='Target', y='Features', ax=ax,hue='Class', dodge=False, palette='Paired')


# <b>Insights from the plots</b>:
# 
# - The two highest correlated feature types are WORST and MEAN and the lowest is the STANDARD ERROR.
# - But now we can't see how the combination of features influence in our target. I am keep them and let my model decide.

# ## Initial Models

# On this section we will:
# - Pick different out-of-the-box models and evaluate them in our training data. My idea here is to try linear, tree based adn ensemble models. Most of the selected models provide feature importance information.
# - See if the first results give us any tips on how to improve our data somehow and test some ideas (feature engineering)
# - Choose the top five most promising and distinct models
# - We have small datasets in our hand. For to get most outof it we use cross-validation technique
# 
# The models we will be using are:
# - Logistic Regression
# - Support Vector Classifier (SVC)
# - Decision Tree
# - Random Forests
# - Gradient Boos Classifier
# - AdaBoost Classifier
# - XGB
# - K-Nearest Neighbors

# In[30]:


# Importing Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Importing other tools
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[31]:


# Defining random seed
seed=123

# Creating Models
logreg = LogisticRegression(solver='lbfgs', random_state=seed)
svc = SVC(random_state=seed, probability=True)
dtree = DecisionTreeClassifier(random_state=seed)
rf = RandomForestClassifier(10, random_state=seed)
gdb = GradientBoostingClassifier(random_state=seed)
adb = AdaBoostClassifier(random_state=seed)
xgb = XGBClassifier(random_state=seed)
knn = KNeighborsClassifier()

first_models = [logreg, svc, dtree, rf, gdb, adb, xgb, knn]
first_model_names = ['Logistic Regression', 'SVC','Decision Tree', 
                     'Random Forest', 'GradientBoosting',
                        'AdaBoost', 'XGB', 'K-Neighbors'] 

# Defining other steps
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, random_state=seed)
std_sca = StandardScaler()


# In[32]:


X_train = train.drop(['diagnosis', 'Target'] ,axis=1)
y_train = train['Target']


# ## Model Evaluation

# ### Choosing the Proper Measure to Evaluate the Model Performance
# There are **a lot** of ways to measure the quality of your model and we must choose it carefully. This is one of the most important parts of a Machine Learning Project.
# 
# Our objective isn't classifying correctly the tumors. If that was the case simply using Accuracy - which is the ratio of correctly predicted classes - would do the job.
# 
# However, the objective of this analysis is **detecting malignant tumors**. And how do we measure that? Not with Accuracy, but with **RECALL**. 
# 
# Recall answers the following question: *from all the malignant tumors in our data, how many did we catch?*. Recall is calculated by dividing the True positives by the total number of positives (positive = malignant). It is important to realize that a high Recall doesn't mean a high Accuracy and there is often a trade-off between different performance measures. 
# 
# That said, we will be making our decisions based on Recall but we will also measure Accuracy to see the difference between them. Moving on!
# 
# <img src="Precision_recall.png" width="500" height="300"/>

# ### Coding Explanation:
# 
# The code on the cell below does the following steps:
# * Setting up:
#     1. Creates an array to store the out-of-fold predictions that we will use later on. Its shape is the training size by the number of models we have;
#     2. Creates a list to store the Accuracy and Recall scores
# * Outer Loop: Iterating through Models
#     1. Creates a data pipeline with the scaler and the model
#     - Creates two arrays to store each fold's accuracy and recall
#     - Executes the inner loop
#     - By the end of the cross-validation, stores the mean and the standard deviation for those two measures in the scores list
# * Inner Loop: Cross-Validation
#     1. Splits the training data into train/validation data
#     2. Fits the model with the CV training data and predicts the validation data
#     3. Stores the out-of-fold predictions (which is the validation predictions) in oof_preds
#     4. Measures the Accuracy and Recall for the fold and stores in an array

# In[33]:


def initial_model(X_train,y_train):
    
    train_size = X_train.shape[0]
    n_models = len(first_models)
    oof_pred = np.zeros((train_size, n_models))
    scores = []

    for n, model in enumerate(first_models):
        model_pipeline = Pipeline(steps=[('Scaler', std_sca),
                                         ('Estimator', model)])
        accuracy = np.zeros(n_folds)
        recall = np.zeros(n_folds)

        for i, (train_ix, val_ix) in enumerate(skf.split(X_train, y_train)):
            x_tr,  y_tr  = X_train.iloc[train_ix], y_train.iloc[train_ix]
            x_val, y_val = X_train.iloc[val_ix],   y_train.iloc[val_ix]

            model_pipeline.fit(x_tr, y_tr)
            val_pred = model_pipeline.predict(x_val)

            oof_pred[val_ix, n] = model_pipeline.predict_proba(x_val)[:,1]

            fold_acc = accuracy_score(y_val, val_pred)
            fold_rec = recall_score(y_val, val_pred)

            accuracy[i] = fold_acc
            recall[i] = fold_rec

        scores.append({'Accuracy'          : accuracy.mean(),
                       'Recall'            : recall.mean()})
    return scores,oof_pred


# ### Initial Model Evaluation

# In[34]:


scores,oof_pred = initial_model(X_train,y_train)


# In[35]:


measure_cols = ['Accuracy', 'Recall']

first_scores = pd.DataFrame(columns=measure_cols)

for name, score in zip(first_model_names, scores):
    
    new_row = pd.Series(data=score, name=name)
    first_scores = first_scores.append(new_row)
    
first_scores = first_scores.sort_values('Recall', ascending=False)
first_scores


# This table shows us each model ordered by its Recall, descending.
# 
# ** Insights **:
# - Logistic Regression and SVC got the highest scores, while KNN and Random Forest the lowest.
# - Most of the got above 95% accuracy and 91% recall on a first try.

# ## Feature Selection

# Most models provide a method that returns feature importances or coefficients so we can have an idea of what is being considered the most important features of our dataset. SVC and KNN are the ones that don't have it.
# 
# Let's see if we can find anything from the other models preferences.

# In[36]:


def feat_imp():
    feature_names = X_train.columns
    feat_imp_df = pd.DataFrame(columns=first_model_names, index=feature_names)

    # Dropping the Models that don't have feature importances for this analysis
    feat_imp_df.drop(['SVC', 'K-Neighbors'], axis=1, inplace=True)

    feat_imp_df['Logistic Regression'] = np.abs(logreg.coef_.ravel())
    feat_imp_df['Decision Tree'] = dtree.feature_importances_
    feat_imp_df['Random Forest'] = rf.feature_importances_
    feat_imp_df['GradientBoosting'] = gdb.feature_importances_
    feat_imp_df['AdaBoost'] = adb.feature_importances_
    feat_imp_df['XGB'] = xgb.feature_importances_
    
    return feat_imp_df


# In[37]:


feat_imp_df = feat_imp()


# In[38]:


feat_imp_df.head()


# > So this is how our table looks like right now. Each model has its own measure for each feature's importances. You will notice that some measures are in different scales. 
# 
# > In order to compare the importances between the models we need to scale them. I will use sklearn MinMaxScaler to shrink them to a [0, 1] interval and then sum the features importances for each model.

# In[39]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

scaled_fi = pd.DataFrame(data=mms.fit_transform(feat_imp_df),
                         columns=feat_imp_df.columns,
                         index=feat_imp_df.index)
scaled_fi['Overall'] = scaled_fi.sum(axis=1)


# In[40]:


scaled_fi.head()


# In[41]:


ordered_ranking = scaled_fi.sort_values('Overall', ascending=False)
fig, ax = plt.subplots(figsize=(10,7), dpi=80)
sns.barplot(data=ordered_ranking, y=ordered_ranking.index, x='Overall', palette='Paired')


# **Insights**:
# - Worst Perimeter is the most important features between models;
# - There is a clear preference for Worst features on models. The top 3 features are 'Worst';
# - From bottom se features are majority.   
# 
# This is what our models have to tell us. If we decided on dropping features based on the correlations plotted we would've gotten some of them wrong. 
# 
# Let's try now removing the Bottom 5 and repeat the training to see if we get any better results. 

# In[42]:


train_v2 = train.drop(ordered_ranking.index[:-6:-1], axis=1)
test_v2 = test.drop(ordered_ranking.index[:-6:-1], axis=1)

X_train_v2 = train_v2.drop(['diagnosis', 'Target'] ,axis=1)
X_test_v2 = test_v2.drop(['diagnosis', 'Target'] ,axis=1)


# In[43]:


scores_v2,oof_pred_v2 =initial_model(X_train_v2,y_train)


# In[44]:


measure_cols = ['Accuracy', 'Recall']

fs_scores = pd.DataFrame(columns=measure_cols)

for name, score in zip(first_model_names, scores_v2):
    
    new_row = pd.Series(data=score, name=name)
    fs_scores = fs_scores.append(new_row)
    
fs_scores = fs_scores.sort_values('Recall', ascending=False)

d={'First Scores':first_scores, 'Less Features':fs_scores}
pd.concat(d, axis=1, sort=False)


# **Insights from Feature Selection**:
# 
# - What changed?
#     - Logistic Regression imporved and leading for now
#     - SVC, AdaBoost and XGB didn't change at all
#     - KNN, Decision Tree, RF and GradientBoosting got worst;
#     
# 
# <b>It is not clear if removing the features was a good decision or not. When in doubt, opt for the simpler choice: We are removing them.</b>
# 

# In[45]:


oof_pred_v2


# In[46]:


oof_dataframe = pd.DataFrame(data=oof_pred_v2, columns=first_model_names, index=train.index)
oof_dataframe['Target'] = train['Target']
#oof_dataframe = oof_dataframe.drop(['LDA', 'Decision Tree', 'Linear SVC'], axis=1)


# In[47]:


oof_dataframe.sample(10)


# Lets see if we can find examples that all models got the classification wrong. The function defined below does just that.

# In[54]:


def all_wrong(x):
    predictions = sum(x[:8])
    target = x[8]
    if (target == 1 and predictions == 0) or        (target == 0 and predictions == 7):
        return True
    
    else: return False


# In[55]:


oof_dataframe['All_wrong'] = round(oof_dataframe).apply(all_wrong, axis=1)
oof_dataframe.query("All_wrong == True")


# ** I decided to tune further top 2 models based on Recall **
# - Logistic Regression
# - SVC
# 

# ** Sklearn's GridSearchCV is our best friend for parameter tuning. **

# In[57]:


from sklearn.model_selection import GridSearchCV

# function for tuning
def train_gridsearch(model, x=X_train_v2, y=y_train, name=None):
    t_model = model
    t_model.fit(x, y)
    print(30*'-')
    if name != None: print(name)
    print('\nBest Parameters:')
    for item in t_model.best_params_.items():
        print(item[0], ': ', item[1])
    print('\nScore: ', t_model.best_score_, '\n')
    print(30*'-')


# In[58]:


from sklearn.base import BaseEstimator, TransformerMixin

class Logger(BaseEstimator, TransformerMixin):
    def __init__(self, apply_log = True):
        self.apply_log = apply_log
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        logX = X.copy()
        
        if self.apply_log:
            logX = np.log1p(X)
            return logX
    
        else: return X

logger = Logger()


# ## Logistic Regression Tuning
# 

# In[59]:


# Logistic Regression Initial Parameters
log_pams = [{'M__solver':['liblinear'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, ],
             'M__penalty':['l1'], 
             'L__apply_log':[True, False]},
            {'M__solver':['lbfgs'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, ],
             'M__penalty':['l2'], 
             'L__apply_log':[True, False]}]

log_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', logreg)])

log_gs = GridSearchCV(log_pipe, log_pams, scoring='recall',cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_gs)


# Our best C is at 10 so we might refine our parameters near that value. A second run on parameter tuning could look like:

# In[61]:


# Logistic Regression Initial Parameters
log_pams = [{'M__solver':['liblinear'],
             'M__class_weight':[None],
             'M__C': [1,2,4, 7, 8],
             'M__penalty':['l1'], 
             'L__apply_log':[False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
log_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', logreg)])

log_gs = GridSearchCV(log_pipe, log_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(log_gs)


# In[62]:


logreg_tuned = log_gs.best_estimator_


# In[64]:


print(confusion_matrix(y_train, log_gs.predict(X_train_v2)))


# In[68]:


sns.heatmap(confusion_matrix(y_train, log_gs.predict(X_train_v2)),
                annot=True, square=True, cbar=False,
                fmt='.0f', cmap='BuGn_r', vmax=10)

print(f'Accuracy:  {100*accuracy_score(y_train, log_gs.predict(X_train_v2)):.4}%     \nRecall: {100*recall_score(y_train, log_gs.predict(X_train_v2)):.4}%')


# In[96]:


#test_v2.head()
y_test = test_v2['Target']

#X_test_scaled = std_sca.transform(X_test_v2)

sns.heatmap(confusion_matrix(y_test, log_gs.predict(X_test_v2)),fmt='.0f',annot=True, square=True, cbar=False,cmap='BuGn_r', vmax=10)

print(f'Accuracy:  {100*accuracy_score(y_test, log_gs.predict(X_test_v2)):.4}%     \nRecall: {100*recall_score(y_test, log_gs.predict(X_test_v2)):.4}%')


# ## SVC Tuning

# In[70]:


# SVC Initial Parameters
svc_pams = [{'M__kernel':['rbf'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.001, 0.01, 0.1, 1, 10, 100, 200],
             'M__gamma':['auto', 'scale', 0.001, 0.01, 0.1],
             'L__apply_log':[True, False]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
svc_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', svc)])

svc_gs = GridSearchCV(svc_pipe, svc_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False, refit=True)

train_gridsearch(svc_gs)


# In[72]:


# SVC Second round Parameters
svc_pams = [{'M__kernel':['rbf'],
             'M__class_weight':[None, 'balanced'],
             'M__C': [0.05, 0.07, 0.1, 0.12, 0.15, 0.2,100,110],
             'M__gamma':[0.01],
             'L__apply_log':[True]}]

# It is important to apply the log transformer before the scaling otherwise we will always get 'number near 0' error.
svc_pipe = Pipeline(steps=[('L', logger),
                           ('S', std_sca),
                           ('M', svc)])

svc_gs = GridSearchCV(svc_pipe, svc_pams, scoring='recall',
                      cv=skf, n_jobs=-1, iid=False)

train_gridsearch(svc_gs)


# In[73]:


sns.heatmap(confusion_matrix(y_train, svc_gs.predict(X_train_v2)),
                annot=True, square=True, cbar=False,
                fmt='.0f', cmap='BuGn_r', vmax=10)

print(f'Accuracy:  {100*accuracy_score(y_train, log_gs.predict(X_train_v2)):.4}%     \nRecall: {100*recall_score(y_train, log_gs.predict(X_train_v2)):.4}%')


# In[97]:


sns.heatmap(confusion_matrix(y_test, svc_gs.predict(X_test_v2)),
                annot=True, square=True, cbar=False,
                fmt='.0f', cmap='BuGn_r', vmax=10)

print(f'Accuracy:  {100*accuracy_score(y_test, svc_gs.predict(X_test_v2)):.4}%     \nRecall: {100*recall_score(y_test, svc_gs.predict(X_test_v2)):.4}%')


# ## Summary - Insights from the results 

# * Logistic Regression Training data summary
#     - Accuracy:  99.5%     
#     - Recall: 98.65%
# 
# * Logistic Regression Testing data summary
#     - Accuracy:  94.15%     
#     - Recall: 92.19%
# 
# * SVC Training data summary
#     - Accuracy:  99.5%     
#     - Recall: 98.65%
# 
# * SVC Training data summary
#     - Accuracy:  95.91%     
#     - Recall: 93.75%

# * Both models got at least 92% recall - for this data this means 5 malignant tumors not detected
# * Our best model SVC performed the best, with **93.75%** malgiinant tumors detected.
