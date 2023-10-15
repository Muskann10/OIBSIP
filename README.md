# Iris Classification

This project demonstrates a simple classification task using the famous Iris dataset. The objective is to classify iris flowers into three species (Setosa, Versicolor, and Virginica) based on four features (sepal length, sepal width, petal length, and petal width).

## Dataset
- The Iris dataset contains 150 samples, with 50 samples for each species.
- The data is stored in a CSV file named `iris.csv`.

## Task 1: Data Exploration and Visualization
- The project starts with data exploration, which includes:
  - Loading the dataset using pandas.
  - Checking the basic information about the data (e.g., data types, null values).
  - Visualizing the data using scatter plots and pair plots.

## Task 2: Data Preprocessing
- Data preprocessing includes:
  - Encoding the target variable (species) into numerical values.
  - Splitting the data into training and testing sets.

## Task 3: Model Building and Evaluation
- The main part of the project is to build and evaluate machine learning models. We use three different algorithms:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Model performance is evaluated using accuracy and a confusion matrix.

## Conclusion
- The project demonstrates how to perform a classification task, from data exploration and preprocessing to model building and evaluation.
- The performance of different algorithms is compared to find the most suitable model for classifying iris flowers.


# Car Price Prediction

This project focuses on predicting the selling price of cars based on various features such as the year, present price, kilometers driven, fuel type, seller type, transmission, and owner history.

## Dataset
- The car price dataset is stored in a CSV file named `car data.csv`.
- It includes information on 301 cars, each described by 9 features.

## Task 1: Data Exploration and Preprocessing
- The project begins with data exploration, which includes:
  - Loading the dataset using pandas.
  - Checking data information, identifying data types, and handling missing values.
  - Encoding categorical features into numerical values.
  - Data visualization using pair plots.

## Task 2: Model Building and Evaluation
- The main part of the project involves building and evaluating machine learning models to predict car prices. We use several algorithms:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Model performance is evaluated using the R-squared score and cross-validation.

## Conclusion
- This project showcases the process of predicting car prices using machine learning.
- Different regression algorithms are applied to the dataset, and their performance is compared.
- The model with the best performance is used to make price predictions.


# Email Spam Detection

This project is focused on the classification of emails as either "spam" or "ham" (non-spam). It uses Natural Language Processing (NLP) techniques to process and classify email text.

## Dataset
- The dataset used for this project is a collection of SMS spam messages. The dataset is stored in a CSV file named `spam.csv`.

## Task 1: Data Preprocessing
- The project begins with data preprocessing, which includes:
  - Loading the dataset using pandas.
  - Renaming columns for clarity.
  - Text preprocessing, tokenization, and removal of stopwords.

## Task 2: Feature Extraction
- Feature extraction is performed using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.

## Task 3: Model Building and Evaluation
- The main part of the project involves building a machine learning model for spam detection using the Multinomial Naive Bayes algorithm.
- Model performance is evaluated based on accuracy, a confusion matrix, and a classification report.

## Conclusion
- This project demonstrates the use of NLP techniques for spam email detection.
- It showcases how to preprocess text data, extract features, and build a classification model to identify spam messages.

