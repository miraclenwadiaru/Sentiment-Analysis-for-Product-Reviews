# Sentiment-Analysis-for-Product-Reviews

## Overview
This is the third project in FLiT project-based learning. I worked with a dataset of product reviews from an e-commerce platform. The dataset includes text reviews and associated ratings, which I used to perform sentiment analysis. In this project, I delved into the world of natural language processing and sentiment analysis.

This repository contains code for sentiment analysis on Amazon product reviews. The analysis involves data preparation, exploratory data analysis (EDA), data preprocessing, text vectorization, model building, and model evaluation.

## Aim
The goal is to build a model that can classify each review as positive or negative based on the context of the text.

## Tools and technologies

+ Streamlit: used for developing a user-friendly and interactive interface for my sentiment analysis application.
+ Github: used for version control and to maintain a centralized repository for my project.
+ Python libraries: used for data cleaning, manipulation, visualization of sentiment distribution, and model building (Pandas, spacy, scikit-learn, matplotlib, etc.)
+ Spyder: developed the code using the Spyder IDE.
+ Jupyter Notebook: used in writing code for analysis and model building.

## Workflow
## Data Preparation
+ Read the dataset using Pandas.
+ I checked for missing values and handled them appropriately.
+ I ensured the dataset was suitable for sentiment analysis.

## Exploratory Data Analysis
+ I explored the dataset to understand the distribution of star ratings.
+ I explored the dataset to understand the distribution of sentiments.
+ I generated a word cloud from my text data to visually identify keywords or patterns in my text data.
+ Made data-driven decisions for preprocessing steps

## Data Preprocessing
+ I cleaned and preprocessed the text data for better model performance.
+ Used the Spacy library to process and analyze text data.
+ Removed irrelevant characters and stopwords, and applied tokenization and lemmatization.

## Model Building
+ Implemented logistic regression and random forest classifier models

## Model Evaluation
+ Evaluated the models using relevant metrics such as accuracy, precision, recall, and F1 score.
+ Compare the performance of the logistic regression model and the random forest classifier model.

## Model Serialization
+ I saved the Random Forest model and vectorizer using the Pickle library. This allows for easy loading and reuse of the trained model without retraining.

## Implementation Details
+ I developed the code using the Spyder IDE.
+ Used Streamlit to create a user-friendly web application for sentiment analysis predictions.

## Model Performance
+ The random forest classifier outperformed the logistic regression model in terms of accuracy.
+ The logistic regression model gave an accuracy score of 88% on the test data, while the random forest classifier model gave an accuracy score of 93% on the test data.

## Getting Started
+ Install the necessary libraries using pip.
+ Download Sentiment Analysis Project ipynb and sentimentapp.py.
+ Run the Python file using Jupyter Notebook and the Python file using Spyder IDE.
+ Open your command prompt and navigate to the directory containing your codes.
+ Run streamlit by writing streamlit run sentimentapp.py. Shortly, a URL will be provided. You can run the app locally on your PC.

## Conclusion
In conclusion, the aim of the project was achieved. The integration of the random forest classifier model with Streamlit provided a user-friendly interface for interpreting customers' sentiments.

However, it is crucial to acknowledge that, like any predictive model, the app’s predictions are not always 100% accurate. While the analysis may offer a meaningful interpretation of customers' sentiment, occasionally misinterpretation may occur. Hence, in the future, there is a need for modification of the app to enhance its accuracy and effectiveness in interpreting customer feedback
