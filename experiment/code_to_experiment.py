from data.process_data import load_data,clean_data
df = load_data('data/disaster_messages.csv','data/disaster_categories.csv')
df = clean_data(df)
df.head()
