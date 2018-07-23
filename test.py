# https://www.kaggle.com/jtrofe/beer-recipes/

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


beer_recipe = pd.read_csv('C:/Users/B-dragon90/Desktop/GitHub/Beer-recipes/recipeData.csv', index_col='BeerID', encoding='latin1')

null_priming = beer_recipe['PrimingMethod'].isnull()

style_cnt = beer_recipe.loc[:,['Style','PrimingMethod']]
style_cnt['NullPriming'] = style_cnt['PrimingMethod'].isnull()
style_cnt['Count'] = 1
style_cnt_grp = style_cnt.loc[:,['Style','Count','NullPriming']].groupby('Style').sum()

style_cnt_grp = style_cnt_grp.sort_values('NullPriming', ascending=False)
style_cnt_grp.reset_index(inplace=True)

def stacked_bar_plot(df, x_total, x_sub_total, sub_total_label, y):
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 8))

    # Plot the total
    sns.set_color_codes("pastel")
    sns.barplot(x=x_total, y=y, data=df, label="Total", color="b")

    # Plot
    sns.set_color_codes("muted")
    sns.barplot(x=x_sub_total, y=y, data=df, label=sub_total_label, color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    sns.despine(left=True, bottom=True)

    return f, ax


def get_sg_from_plato(plato):
    sg = 1 + (plato / (258.6 - ( (plato/258.2) *227.1) ) )
    sg = ((-1) * 616.868) + (1111.14 * sg) - (630.272 * pow(sg, 2)) + (135.997 * pow(sg, 3))
    return sg

beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['OG']) if row['SugarScale'] == 'Plato' else row['OG'], axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['FG']) if row['SugarScale'] == 'Plato' else row['FG'], axis=1)
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['BoilGravity']) if row['SugarScale'] == 'Plato' else row['BoilGravity'], axis=1)



num_feats_list = ['Size(L)', 'OG_sg', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'PitchRate', 'PrimaryTemp']



from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import train_test_split

# 사용할 Feature 설정
features_list= ['StyleID', #target
                'OG_sg','FG_sg','ABV','IBU','Color', #standardized fields
                'SugarScale', 'BrewMethod', #categorical features
                'Size(L)', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency'
                ]
# MashThickness, PitchRate, PrimaryTemp


clf_data = beer_recipe.loc[:, features_list]

# Label encoding
cat_feats_to_use = list(clf_data.select_dtypes(include=object).columns)
for feat in cat_feats_to_use:
    encoder = LabelEncoder()
    clf_data[feat] = encoder.fit_transform(clf_data[feat])

# Fill null values
num_feats_to_use = list(clf_data.select_dtypes(exclude=object).columns)
for feat in num_feats_to_use:
    imputer = Imputer(strategy='most_frequent') #median
    clf_data[feat] = imputer.fit_transform(clf_data[feat].values.reshape(-1,1))

# Seperate Targets from Features
X = clf_data.iloc[:, 1:]
y = clf_data.iloc[:, 0] #the target were the first column I included

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=35)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#다시 무결성 체크
sanity_df = pd.DataFrame(X_train, columns = X.columns)


#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf1 = SVC()
# clf2 = RandomForestClassifier()
# clf3 = LogisticRegression()
clf1.fit(X_train, y_train)
# clf2.fit(X_train, y_train)
# clf3.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred1 = clf1.predict(X_test)
score1 = accuracy_score(y_test, y_pred1)
print('Accuracy: {}'.format(score1))

# y_pred2 = clf2.predict(X_test)
# score2 = accuracy_score(y_test, y_pred2)
# print('Accuracy: {}'.format(score2))
#
# y_pred3 = clf3.predict(X_test)
# score3 = accuracy_score(y_test, y_pred3)
# print('Accuracy: {}'.format(score3))
