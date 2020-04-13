# About Tumors
- Tumors are abnormal growths in your body. They are made up of extra cells. Normally, cells grow and divide to form new cells as your body needs them. When cells grow old, they die, and new cells take their place. Sometimes, this process goes wrong. New cells form when your body does not need them, and old cells do not die when they should. When these extra cells form a mass, it is called a tumor.

- Tumors can be either benign or malignant. Benign tumors aren't cancer. Malignant ones are. Benign tumors grow only in one place. They cannot spread or invade other parts of your body. Even so, they can be dangerous if they press on vital organs, such as your brain.

## key differences between benign and malignant tumors?
<img src="benign_malignant.png" />

### Problem solving approach

This dataset contains information on 569 breast tumors and the mean, standard error and worst measures for 10 different properties. I start with an EDA analysing each properties' distribution, followed by the pair interactions and then the correlations with our target.

After the EDA I set up 8 out-of-the-box models for a first evaluation and use stratified cross-validation to measure them. I use **Recall** instead of **Accuracy or F1-Score** since I want to detect all malignant tumors. 

After the first results I analyse features importances, do a single round of feature selection and evaluate the models again. At the end I analyse model errors from the 8 first models I choosen 5 models for fine tuning: 
**Logistic Regression, SVC, Random Forest, Gradient Boosting and KNN.**

Then I proceed to tune the top 2 models using **GridSearchCV** and prepare the data for model by predicting probabilities for both train and test sets. 


## Dataset General Information

- Original format: csv
- Dataset shape: 569 x 33 (rows x columns)
- There are no null values in this data.
- The values are in different scales

### Features information

For each sample ten properties were measured:

<ol>
    <li><b>Radius</b> - Mean distances from center to points on the perimeter</li>
    <li><b>Texture</b> - Standard deviation of gray scale values</li>
    <li><b>Perimeter</b></li>
    <li><b>Area</b></li>
    <li><b>Smoothness</b> - Local variation in radius lengths</li>
    <li><b>Compactness</b> - Perimeter^2/Area - 1</li>
    <li><b>Concavity</b> - Severity of concave portions of the contour</li>
    <li><b>Concave points</b> - Number of concave portions of the contour</li>
    <li><b>Simmetry</b></li>
    <li><b>Fractal Dimension</b> - Coastline approximation - 1 </li>
</ol>


And for each of these properties we have three calculated values:
- **Mean**
- **Standard Error**
- **Worst** (Average of the 3 largest values)

All the measures are float types.

### Target

Our target is the categorical column either B (benign) or M (malignant).<br>
There are 357 benign classes and 212 malignant classes - roughly **37% malignant tumors**.


## Model Evaluation

### Choosing the Proper Measure to Evaluate the Model Performance
There are **a lot** of ways to measure the quality of your model and we must choose it carefully. This is one of the most important parts of a Machine Learning Project.

Our objective isn't classifying correctly the tumors. If that was the case simply using Accuracy - which is the ratio of correctly predicted classes - would do the job.

However, the objective of this analysis is **detecting malignant tumors**. And how do we measure that? Not with Accuracy, but with **RECALL**. 

Recall answers the following question: *from all the malignant tumors in our data, how many did we catch?*. Recall is calculated by dividing the True positives by the total number of positives (positive = malignant). It is important to realize that a high Recall doesn't mean a high Accuracy and there is often a trade-off between different performance measures. 

That said, we will be making our decisions based on Recall but we will also measure Accuracy to see the difference between them. Moving on!

<img src="Precision_recall.png" width="500" height="300"/>



### Coding Explanation:

The code on the cell below does the following steps:
* Setting up:
    1. Creates an array to store the out-of-fold predictions that we will use later on. Its shape is the training size by the number of models we have;
    2. Creates a list to store the Accuracy and Recall scores
* Outer Loop: Iterating through Models
    1. Creates a data pipeline with the scaler and the model
    - Creates two arrays to store each fold's accuracy and recall
    - Executes the inner loop
    - By the end of the cross-validation, stores the mean and the standard deviation for those two measures in the scores list
* Inner Loop: Cross-Validation
    1. Splits the training data into train/validation data
    2. Fits the model with the CV training data and predicts the validation data
    3. Stores the out-of-fold predictions (which is the validation predictions) in oof_preds
    4. Measures the Accuracy and Recall for the fold and stores in an array
    
## Summary - Insights from the results 

* Logistic Regression on training data
    - Accuracy:  99.5%     
    - Recall: 98.65%

* Logistic Regression on testing data
    - Accuracy:  94.15%     
    - Recall: 92.19%

* SVC on training data
    - Accuracy:  99.5%     
    - Recall: 98.65%

* SVC on testing data
    - Accuracy:  95.91%     
    - Recall: 93.75%
    
* Both models got at least 92% recall - for this data this means 5 malignant tumors not detected
* Our best model SVC performed the best, with **93.75%** malgiinant tumors detected.


    
