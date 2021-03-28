# Credit Risk Using Machine Leaning
 
![Credit Risk](Images/credit-risk.jpg)

## Background

This project aims to help the investors mitigate risk by predicting credit risk with machine learning techniques.

I  have built and evaluated several machine learning models to predict credit risk using data would typically come from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so I have employed different techniques for training and evaluating models with imbalanced classes. I have used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

- - -
## Resampling

Used the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data and built and evaluated logistic regression classifiers using the resampled data.

To begin:

1. Read the CSV into a DataFrame.

2. Split the data into Training and Testing sets.

3. Scaled the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.

4. Used the provided code to run a Simple Logistic Regression:
    * Fit the `logistic regression classifier`.
    * Calculate the `balanced accuracy score`.
    * Display the `confusion matrix`.
    * Print the `imbalanced classification report`.

Next:

1. Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms.

2. Undersample the data using the `Cluster Centroids` algorithm.

3. Over- and undersample using a combination `SMOTEENN` algorithm.


For each of the above, I have:

1. Trained a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

2. Calculated the `balanced accuracy score` from `sklearn.metrics`.

3. Displayed the `confusion matrix` from `sklearn.metrics`.

4. Printed the `imbalanced classification report` from `imblearn.metrics`.


I have used this to answer the following questions:

* Which model had the best balanced accuracy score?
>Naive Random Oversampling, SMOTE Oversampling and SMOTEEN have the same and the highest balanced accuracy score
* Which model had the best recall score?
>All the models have the same recall score of 0.99
* Which model had the best geometric mean score?
>All the models have the same geometric mean score of 0.99

#### Ensemble Learning

Here, first I train and compare two different ensemble classifiers to predict loan risk and evaluate each model. I have used the [Balanced Random Forest Classifier](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html#imblearn-ensemble-balancedrandomforestclassifier) and the [Easy Ensemble Classifier](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.EasyEnsembleClassifier.html#imblearn-ensemble-easyensembleclassifier).

To begin:

1. Read the data into a DataFrame using the provided starter code.

2. Split the data into training and testing sets.

3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.


Next:

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.

2. Calculate the balanced accuracy score from `sklearn.metrics`.

3. Display the confusion matrix from `sklearn.metrics`.

4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.

5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.


I have used this to answer the following questions:

* Which model had the best balanced accuracy score?
>Easy ensemble classifier
* Which model had the best recall score?
>Easy ensemble classifier
* Which model had the best geometric mean score?
>Easy ensemble classifier
* What are the top three features?
>Total Rec Prncp, Total Pymnt, Total Pyment Inv