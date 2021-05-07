import pandas as pd
import importlib
import sys

for module in sys.modules.values():
    importlib.reload(module)
from data.process_data import load_data,clean_data,save_data


df = load_data('data/disaster_messages.csv','data/disaster_categories.csv')
df = clean_data(df)
save_data(df,'data/DisasterResponse.db')
df.head()
