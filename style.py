import os
import pandas as pd

beer_recipe = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Beer-recipes/recipeData_cleansing.csv', index_col='BeerID', encoding='latin1')
beer_recipe.columns.values[2] = "Size"

folder = 'C:/Users/paramount/Desktop/Github/Beer-recipes'

path_mean = os.path.join(folder, 'beer_recipe_mean.csv')
beer_recipe_mean = beer_recipe.groupby(['StyleID'], as_index=True).mean()
beer_recipe_mean.to_csv(path_mean)

path_std = os.path.join(folder, 'beer_recipe_std.csv')
beer_recipe_std = beer_recipe.groupby(['StyleID'], as_index=True).std()
beer_recipe_std.to_csv(path_std)
