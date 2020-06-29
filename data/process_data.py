import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, on='id',  how='outer')
    return df

def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    row = row.str[:-2]
    category_colnames = row.unique()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].replace(2,1)
    df = df.drop(['categories'], axis = 1)
    df_new = pd.concat([df, categories], axis=1, sort=False)
    df_new = df_new.drop_duplicates(subset=None, keep='first', inplace=False)
    return df_new
    
def save_data(df, database_filename):
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('table', engine, index=False)
    pass  


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
    #messages_filepath = 'data/disaster_messages.csv'
    #categories_filepath = 'data/disaster_categories.csv'
    #database_filename = 'data/DisasterResponse.db'
    main()