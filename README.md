# Disaster Response Pipeline Project

## Summary

### Overview
In this project, students'll apply data engineer skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
I use a data set containing real messages that were sent during disaster events. And I created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency. This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### Content Explaination
There are three folders: app, data, models. 

#### 1. data folder - to create a ETL pipeline
Overview:  The first part of my data pipeline is the Extract, Transform, and Load process. Here, I read the dataset, clean the data, and then store it in a SQLite database. I did the data cleaning with pandas. To load the data into an SQLite database, I used the pandas dataframe .to_sql() method and an SQLAlchemy engine.

1.1 disaster_categories.csv
This dataset contains the 36 category type, one messages can be categorized as multiple categories at the same time. 

1.2 disaster_messages.csv
This dataset contains original messages and genre of messages. 

1.3 process_data.py
In this Python script, I write a data cleaning pipeline that:
	- Loads the messages and categories datasets
	- Merges the two datasets
	- Cleans the data
	- Stores it in a SQLite database

1.4 DisasterResponse.db
This database is the output of process_data.py, then I will use this database to build a machine learning pipeline

#### 2. model folder - to create a machine learning pipeline

Overview:  For the machine learning portion, I split the data into a training set and a test set. Then, I created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, I export model to a pickle file. 

2.1 train_classifier.py
In this Python script, I write a machine learning pipeline that:
	- Loads data from the SQLite database
	- Splits the dataset into training and test sets
	- Builds a text processing and machine learning pipeline
	- Trains and tunes a model using GridSearchCV
	- Outputs results on the test set
	- Exports the final model as a pickle file

2.2 classifier.pkl
This pickle file is the output of train_classifier.py. 

#### 3. app folder - to display result through Flask Web App

Overview: In the last step, I displayed my results in a Flask web app. I uploaded my database file and pkl file with my model. This is the part of the project that allows for the most creativity. 

3.1 run.py
When we run this file will see that the web app already works and displays a visualization. 

3.2 templates folder
Contains the supportive documents and link to run the run.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
