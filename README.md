# Disaster Response Pipeline Project

This project contains the Code to complete the Disaster Pipeline Data Engineering Project of the Udacity Nanodegree Data Scientist. The goal is to cluster messages into different categories. 

All required packages are listed in the requirements file. First move to the project as a root directory and then call the following commands.

The first two steps are to clean the data and fit a model on the cleaned dataset. For this purpose a XGBoost model is used with a RandomizedSearchCV grid to optimize hyperparameters.

# File Structure

## configuration 
- file requirements.txt contains the necessary packages needed


## ETL 
- all files and scripts in subfolder data
- raw data in data/disaster_categories.csv and data/disaster_messages.csv
- processing script in process_data.py 
- preprocessed data is saved in DiasterResponse.db

## model training
- all files and scripts in subfolder models 
- train_classifier.py is the script which should be used for training (train_classifier.ipynb was my script to experiment in a notebook)
- classifier (whole pipeline) is saved as classifier.pkl

## app
- all files and scripts in subfolder app 
- run.py contains the script to run app (see instructions below how to start it)
- templates/master.html contains the master template with the HTML code vor the dashboard
- templates/go.html is the code which is displayed after a tweet is analyzed

## pictures
- contains screenshots of the app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

