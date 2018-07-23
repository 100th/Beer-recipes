import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


beer_recipe = pd.read_csv('C:/Users/paramount/Desktop/GitHub/Beer-recipes/recipeData.csv', index_col='BeerID', encoding='latin1')
#beer_recipe.head()


# print(beer_recipe.info(verbose=False))




# Null 값이 얼마나 많은지
# %matplotlib inline
# msno.matrix(beer_recipe.sample(1000))





null_priming = beer_recipe['PrimingMethod'].isnull()
# print('PrimingMethod is null on {} rows out of {}, so {} % of the time'.format(null_priming.sum(), len(beer_recipe), round((null_priming.sum()/len(beer_recipe))*100,2)))







style_cnt = beer_recipe.loc[:,['Style','PrimingMethod']]
style_cnt['NullPriming'] = style_cnt['PrimingMethod'].isnull()
style_cnt['Count'] = 1
style_cnt_grp = style_cnt.loc[:,['Style','Count','NullPriming']].groupby('Style').sum()

style_cnt_grp = style_cnt_grp.sort_values('NullPriming', ascending=False)
style_cnt_grp.reset_index(inplace=True)





# 그래프 그리는 함수
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

# f, ax = stacked_bar_plot(style_cnt_grp[:20], 'Count', 'NullPriming', 'Priming Method is null', 'Style')
# ax.set(title='Missing Values in PrimingMethod column, per style', ylabel='', xlabel='Count of Beer Recipes')
# sns.despine(left=True, bottom=True)



# print( list(beer_recipe.select_dtypes(include=object).columns))
#
# print(beer_recipe.PrimingAmount.unique())


# ax = sns.countplot(x='SugarScale', data=beer_recipe)
# ax.set(title='Frequency table of possible values in SugarScale')
# sns.despine(left=True, bottom=True)
#
# print('SugarScale has {} null values'.format(beer_recipe.SugarScale.isnull().sum()))


# ax = sns.countplot(x='BrewMethod', data=beer_recipe)
# ax.set(title='Frequency table of possible values in BrewMethod')
# sns.despine(left=True, bottom=True)
#
# print('BrewMethod has {} null values'.format(beer_recipe.BrewMethod.isnull().sum()))




# print('PrimingMethod has {} unique values'.format(beer_recipe.PrimingMethod.nunique()))
# print(beer_recipe.PrimingMethod.unique()[:20])



# print( list( beer_recipe.select_dtypes(exclude=object)))




# 공식
def get_sg_from_plato(plato):
    sg = 1 + (plato / (258.6 - ( (plato/258.2) *227.1) ) )
    sg = ((-1) * 616.868) + (1111.14 * sg) - (630.272 * pow(sg, 2)) + (135.997 * pow(sg, 3))
    return sg

beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['OG']) if row['SugarScale'] == 'Plato' else row['OG'], axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['FG']) if row['SugarScale'] == 'Plato' else row['FG'], axis=1)
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['BoilGravity']) if row['SugarScale'] == 'Plato' else row['BoilGravity'], axis=1)




# count, mean, std, min, 25%, 50%, 75%, max 구하기
num_feats_list = ['Size(L)', 'OG_sg', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'PitchRate', 'PrimaryTemp']
beer_recipe.loc[:, num_feats_list].describe().T

"""

# 지표값 범위로 분류하기
vlow_scale_feats = ['OG_sg', 'FG_sg', 'BoilGravity_sg', 'PitchRate']
low_scale_feats = ['ABV', 'MashThickness']
mid_scale_feats = ['Color', 'BoilTime', 'Efficiency', 'PrimaryTemp']
high_scale_feats = ['IBU', 'Size(L)',  'BoilSize']

# vlow_scale_feats
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, vlow_scale_feats], orient='h')
ax.set(title='Boxplots of very low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

# low_scale_feats
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, low_scale_feats], orient='h')
ax.set(title='Boxplots of low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

#mid_scale_feats
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, mid_scale_feats], orient='h')
ax.set(title='Boxplots of medium scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

#high_scale_feats
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, high_scale_feats], orient='h')
ax.set(title='Boxplots of high scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

"""

# 몇 가지 종류가 있냐
# print('There are {} different styles of beer'.format(beer_recipe.StyleID.nunique()))

"""
# 원 그래프 그리기
# Get top10 styles
top10_style = list(style_cnt_grp['Style'][:10].values)

# Group by current count information computed earlier and group every style not in top20 together
style_cnt_other = style_cnt_grp.loc[:, ['Style','Count']]
style_cnt_other.Style = style_cnt_grp.Style.apply(lambda x: x if x in top10_style else 'Other')
style_cnt_other = style_cnt_other.groupby('Style').sum()

# Get ratio of each style
style_cnt_other['Ratio'] = style_cnt_other.Count.apply(lambda x: x/float(len(beer_recipe)))
style_cnt_other = style_cnt_other.sort_values('Count', ascending=False)

f, ax = plt.subplots(figsize=(8, 8))
explode = (0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0)
plt.pie(x=style_cnt_other['Ratio'], labels=list(style_cnt_other.index), startangle = 180, autopct='%1.1f%%', pctdistance= .9, explode=explode)
plt.title('Ratio of styles across dataset')
plt.show()
"""

# 막대 그래프 그리기
#plt.barh(list(style_cnt_other.index), style_cnt_other['Count'])
# style_cnt_other['Ratio'].plot(kind='barh', figsize=(12,6),)
# plt.title('Ratio of styles across dataset')
# sns.despine(left=True, bottom=True)
# plt.gca().invert_yaxis()

"""
# 상관관계 5x5 그림. 우리가 관심있는 것만 선택해서
pairplot_df = beer_recipe.loc[:, ['Style','OG_sg','FG_sg','ABV','IBU','Color']]

# create the pairplot
sns.set(style="dark")
sns.pairplot(data=pairplot_df)
plt.show()
"""


# Outlier 찾는 이상한 팽이 모양 그래프 그리기
# style_cnt_grp = style_cnt_grp.sort_values('Count', ascending=False)
top5_style = list(style_cnt_grp['Style'][:5].values)
#
# top5_style_df = pairplot_df[pairplot_df['Style'].isin(top5_style)]
#
# f, ax = plt.subplots(figsize=(12, 8))
# sns.violinplot(x='Style', y='OG_sg',data=top5_style_df)
# plt.show()


# 실패한 선형 관계 그래프 (그래프가 두 개 나옴)
# # Get Top5 styles
# top5_style = list(style_cnt_grp['Style'][:5].values)
# beer_recipe['Top5_Style'] = beer_recipe.Style.apply(lambda x: x if x in top5_style else 'Other')
#
# # Create Reg plot
# sns.lmplot(x='ABV', y='OG', hue='Top5_Style', col='Top5_Style', col_wrap=3, data=beer_recipe, n_boot=100)

"""
# 정상적인 선형 그래프
# Create Reg plot
sns.lmplot(x='ABV', y='OG_sg', hue='Top5_Style', col='Top5_Style', col_wrap=3, data=beer_recipe, n_boot=100)
"""


###############################################################
# imports
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



# 무결성 확인. null값 남아있는지 확인.
X.info()


# 스케일링 (Scaling)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#다시 무결성 체크
sanity_df = pd.DataFrame(X_train, columns = X.columns)
sanity_df.describe().T



# 알고리즘 import
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf1 = SVC()
clf2 = RandomForestClassifier()
clf3 = LogisticRegression()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred1 = clf1.predict(X_test)
score1 = accuracy_score(y_test, y_pred1)
print('Accuracy: {}'.format(score1))

y_pred2 = clf2.predict(X_test)
score2 = accuracy_score(y_test, y_pred2)
print('Accuracy: {}'.format(score2))

y_pred3 = clf3.predict(X_test)
score3 = accuracy_score(y_test, y_pred3)
print('Accuracy: {}'.format(score3))


"""
# 어떤 Feature가 영향력 있니????
feats_imp = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['FeatureImportance'])
feats_imp = feats_imp.sort_values('FeatureImportance', ascending=False)

feats_imp.plot(kind='barh', figsize=(12,6), legend=False)
plt.title('Feature Importance from RandomForest Classifier')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()
"""
