import numpy as np
import pandas as pd

table = pd.read_csv('C:/Users/B-dragon90/Desktop/GitHub/Beer-recipes/test.csv', index_col='ID', encoding='latin1')
table.head()

table.loc[table.year >= 2001, 'year'] = None


table.head()

table.isnull()
