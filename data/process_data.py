import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

def load_data(messages_filepath, categories_filepath):
    '''
    Import and merge csv files

    messages_filepath: path to messages file
    categories_filepath: path to categories file

    returns merged file
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id")

    return df

def clean_data(df):
    '''
    clean dataframe

    df: merged dataframe containing messages and categories

    returns cleaned dataframe
    '''
    categories = df.categories.str.split(";",expand=True)

    # extract column names out of values
    row = categories.loc[0,:]
    category_colnames = row.map(lambda x: str(x)[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # remove column name from value and change to numeric
    for column in categories:
       # set each value to be the last character of the string
       categories[column] = categories[column].map(lambda x: str(x)[-1])

       # convert column from string to binary (few values with >0 are converted to 1)
       categories[column] = pd.to_numeric(categories[column])
       categories[column] = categories[column].map(lambda x: 1 if x > 0 else 0)

    # drop original categories column
    df = df.drop(['categories'],axis=1)

    # concate new categories to dataframe
    df = pd.concat([df,categories],axis=1)

    return df 


def save_data(df, database_filename):
    '''
    save cleaned dataframe to sql database

    df: dataframe to save
    database_filename: filename of the database to be saved to
    '''
    database_name = 'sqlite:///' + database_filename
    engine = create_engine(database_name)
    df.to_sql('disaster_clean', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
