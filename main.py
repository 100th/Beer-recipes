# https://www.kaggle.com/jtrofe/beer-recipes/

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


# 데이터 head 보기
beer_recipe = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Beer-recipes/recipeData_cleansing2.csv', index_col='BeerID', encoding='latin1')
beer_recipe.columns.values[2] = "Size"
beer_recipe.head()


# 데이터가 어떻게 이루어져있는지
print(beer_recipe.info(verbose=False))


# Null 값이 얼마나 많은지
%matplotlib inline
msno.matrix(beer_recipe.sample(1000))


# PrimingMethod의 Null이 얼마나 많은지
# null_priming = beer_recipe['PrimingMethod'].isnull()
# print('PrimingMethod is null on {} rows out of {}, so {} % of the time'.format(null_priming.sum(), len(beer_recipe), round((null_priming.sum()/len(beer_recipe))*100,2)))


# BrewMethod에 Null이 얼마나 많은지
null_brew = beer_recipe['BrewMethod'].isnull()
print('BrewMethod is null on {} rows out of {}, so {} % of the time'.format(null_brew.sum(), len(beer_recipe), round((null_brew.sum()/len(beer_recipe))*100,2)))


# 그룹화
style_cnt = beer_recipe.loc[:,['StyleID']]
print(style_cnt.head())
style_cnt['Count'] = 1
style_cnt_grp = style_cnt.loc[:,['StyleID','Count']].groupby('StyleID').sum()
print(style_cnt_grp)
style_cnt_grp = style_cnt_grp.sort_values('Count', ascending=False)
style_cnt_grp.reset_index(inplace=True)
print(style_cnt_grp)


# 그래프 그리는 함수 정의 -> PrimingMethod Missing Values 찾는거라 여기서 쓰고 끝임 -> 쓸모 없음
# def stacked_bar_plot(df, x_total, x_sub_total, sub_total_label, y):
#     # Initialize the matplotlib figure
#     f, ax = plt.subplots(figsize=(12, 8))
#
#     # Plot the total
#     sns.set_color_codes("pastel")
#     sns.barplot(x=x_total, y=y, data=df, label="Total", color="b")
#
#     # Plot
#     sns.set_color_codes("muted")
#     sns.barplot(x=x_sub_total, y=y, data=df, label=sub_total_label, color="b")
#
#     # Add a legend and informative axis label
#     ax.legend(ncol=2, loc="lower right", frameon=True)
#     sns.despine(left=True, bottom=True)
#
#     return f, ax


# 전체 대비 PrimingMethod Missing Value 비율 나타내는 그래프 -> PrimingMethod는 못쓴다고 봐야한다. 너무 높음.
# f, ax = stacked_bar_plot(style_cnt_grp[:20], 'Count', 'NullPriming', 'Priming Method is null', 'Style')
# ax.set(title='Missing Values in PrimingMethod column, per style', ylabel='', xlabel='Count of Beer Recipes')
# sns.despine(left=True, bottom=True)


# Feature에 뭐가 있는지 확인 ########### 이거 object라 안믿는게
# print( list(beer_recipe.select_dtypes(include=object).columns))


# PrimingAmount에 뭐가 들었는지 확인 -> 이 것도 버려야 한다.
# print(beer_recipe.PrimingAmount.unique())


# SugarScale에 뭐가 몇 개 들어있는지 Count
beer_recipe.loc[beer_recipe.SugarScale == 'All Grain', 'SugarScale'] = None
beer_recipe.loc[beer_recipe.SugarScale == 'BIAB', 'SugarScale'] = None
beer_recipe.loc[beer_recipe.SugarScale == 'Partial Mash', 'SugarScale'] = None
beer_recipe.loc[beer_recipe.SugarScale == 'extract', 'SugarScale'] = None
ax = sns.countplot(x='SugarScale', data=beer_recipe)
ax.set(title='Frequency table of possible values in SugarScale')
sns.despine(left=True, bottom=True)
print('SugarScale has {} null values'.format(beer_recipe.SugarScale.isnull().sum()))


# BrewMethod에 뭐가 몇 개 들어있는지 Count
ax = sns.countplot(x='BrewMethod', data=beer_recipe)
ax.set(title='Frequency table of possible values in BrewMethod')
sns.despine(left=True, bottom=True)
print('BrewMethod has {} null values'.format(beer_recipe.BrewMethod.isnull().sum()))


# PrimingMethod가 몇 개의 Unique한 값을 가지는지 -> 이거 버려야 하는데 왜 자꾸 가져가는지 모르겠다.
#print('PrimingMethod has {} unique values'.format(beer_recipe.PrimingMethod.nunique()))


# PrimingMethod가 가진 Unique한 값을 보여줌
#print(beer_recipe.PrimingMethod.unique()[:20])


# Feature에 뭐가 있는지 확인
print( list( beer_recipe.select_dtypes(exclude=object)))


# 공식 구하는 함수
def get_sg_from_plato(plato):
    sg = 1 + (plato / (258.6 - ( (plato/258.2) *227.1) ) )
    # plato = ((-1) * 616.868) + (1111.14 * sg) - (630.272 * pow(sg, 2)) + (135.997 * pow(sg, 3))
    return sg


# 공식을 이용하여 OG_sg, FG_sg, BoilGravity_sg를 구한다.
beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['OG']) if row['SugarScale'] == 'Plato' else row['OG'], axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['FG']) if row['SugarScale'] == 'Plato' else row['FG'], axis=1)
beer_recipe.loc[beer_recipe.BoilGravity == 0, 'BoilGravity'] = None
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['BoilGravity']) if row['SugarScale'] == 'Plato' else row['BoilGravity'], axis=1)


# 데이터의 count, mean, std, min, 25%, 50%, 75%, max 구하기
num_feats_list = ['Size', 'OG', 'OG_sg', 'FG', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'BrewMethod']
beer_recipe.loc[:, num_feats_list].describe().T


# 지표값 범위로 분류하기
vlow_scale_feats = ['OG_sg', 'FG_sg', 'BoilGravity_sg']
low_scale_feats = ['ABV', 'MashThickness', 'BrewMethod']
mid_scale_feats = ['Color', 'BoilTime', 'Efficiency']
high_scale_feats = ['IBU', 'Size', 'BoilSize']

# vlow_scale_feats
beer_recipe.loc[beer_recipe.OG_sg > 1.057 + 1.5 * (1.067 - 1.050), 'OG_sg'] = None
beer_recipe.loc[beer_recipe.FG_sg > 1.013 + 1.5 * (1.016 - 1.011), 'FG_sg'] = None
# beer_recipe.loc[beer_recipe.BoilGravity_sg >= 1.1, 'BoilGravity_sg'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, vlow_scale_feats], orient='h')
ax.set(title='Boxplots of very low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

# low_scale_feats
beer_recipe.loc[beer_recipe.ABV > 5.8 + 1.5 * (6.83 - 5.09), 'ABV'] = None
# beer_recipe.loc[beer_recipe.MashThickness >= 80, 'MashThickness'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, low_scale_feats], orient='h')
ax.set(title='Boxplots of low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

#mid_scale_feats
beer_recipe.loc[beer_recipe.Color > 80, 'Color'] = None
# beer_recipe.loc[beer_recipe.BoilTime >= 120, 'BoilTime'] = None
# beer_recipe.loc[beer_recipe.BoilTime == 0, 'BoilTime'] = None
# beer_recipe.loc[beer_recipe.Efficiency == 0, 'Efficiency'] = None
# beer_recipe.loc[beer_recipe.PrimaryTemp >= 100, 'PrimaryTemp'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, mid_scale_feats], orient='h')
ax.set(title='Boxplots of medium scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

#high_scale_feats
beer_recipe.loc[beer_recipe.IBU > 200, 'IBU'] = None
# beer_recipe.loc[beer_recipe.Size >= 100, 'Size'] = None
# beer_recipe.loc[beer_recipe.BoilSize >= 100, 'BoilSize'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, high_scale_feats], orient='h')
ax.set(title='Boxplots of high scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)


# 몇 가지 종류의 맥주가 있냐 (StyleId로 판단)
print('There are {} different styles of beer'.format(beer_recipe.StyleID.nunique()))


# 원 그래프 그리기


top10_style = list(style_cnt_grp['StyleID'][:10].values)
print(top10_style)


# StyleID Top10만 남긴다
# beer_recipe = beer_recipe[beer_recipe.StyleID != 2]



style_cnt_other = style_cnt_grp.loc[:, ['StyleID','Count']]
style_cnt_other.StyleID = style_cnt_grp.StyleID.apply(lambda x: x if x in top10_style else 'Other')
style_cnt_other = style_cnt_other.groupby('StyleID').sum()
style_cnt_other['Ratio'] = style_cnt_other.Count.apply(lambda x: x/float(len(beer_recipe)))
style_cnt_other = style_cnt_other.sort_values('Count', ascending=False)
f, ax = plt.subplots(figsize=(8, 8))
plt.pie(x=style_cnt_other['Ratio'], labels=list(style_cnt_other.index), startangle = 180, autopct='%1.1f%%', pctdistance= .9)
plt.title('Ratio of styles across dataset')
plt.show()


beer_recipe.head()












# 상관관계 그림. 우리가 관심있는 것만 선택해서
pairplot_df = beer_recipe.loc[:, ['Efficiency', 'MashThickness']]    #'StyleID', 'Size', 'OG_sg', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'BrewMethod'
len(pairplot_df)
pairplot_df2 = pairplot_df.dropna()     # dropna 하면 69656개 -> 41900개
len(pairplot_df2)
sns.set(style="dark")
sns.pairplot(data=pairplot_df2)
plt.show()


# Outlier 찾는 이상한 팽이 모양 그래프 그리기
style_cnt_grp = style_cnt_grp.sort_values('Count', ascending=False)
top5_style = list(style_cnt_grp['StyleID'][:5].values)
top5_style_df = pairplot_df[pairplot_df['StyleID'].isin(top5_style)]
f, ax = plt.subplots(figsize=(12, 8))
sns.violinplot(x='StyleID', y='OG_sg',data=top5_style_df)
plt.show()


# 선형 관계 그래프
top5_style = list(style_cnt_grp['StyleID'][:5].values)
beer_recipe['Top5_Style'] = beer_recipe.StyleID.apply(lambda x: x if x in top5_style else 'Other')
sns.lmplot(x='ABV', y='OG_sg', hue='Top5_Style', col='Top5_Style', col_wrap=3, data=beer_recipe, n_boot=100)


# 사용할 Feature 설정
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import train_test_split

features_list= ['StyleID', #target
                'ABV','IBU','Color', 'OG_sg','FG_sg', 'Size',
                'BoilSize', 'Efficiency', 'BoilGravity_sg', 'BrewMethod']
clf_data = beer_recipe.loc[:, features_list]
len(clf_data)

# 결측치 제거 두 가지 방법
# 1. 하나라도 Null이 있으면 그 행 제거.     2. 평균으로 채워넣기
include_object_list = []
clf_data2 = beer_recipe.loc[:, include_object_list]
exclude_object_list = ['StyleID', 'ABV', 'IBU', 'Color', 'OG_sg','FG_sg', 'Size', 'BoilSize', 'Efficiency', 'BoilGravity_sg', 'BrewMethod']
clf_data3 = beer_recipe.loc[:, exclude_object_list]

len(clf_data)
clf_data2 = clf_data2.dropna()     # clf_data2는 문자라서 dropna() 썼다.
clf_data3 = clf_data3.fillna(clf_data3.mean())   # clf_data3는 숫자라서 평균으로 채웠다.

len(clf_data2)
len(clf_data3)
# str 형식으로 나오는 Feature (SugarScale) NULL 값 채우고 인코딩 = 숫자로 나타낸다.
cat_feats_to_use = list(clf_data2.select_dtypes(include=object).columns)
for feat in cat_feats_to_use:
    encoder = LabelEncoder()
    clf_data2[feat] = encoder.fit_transform(clf_data2[feat])

# float 형식 Feature의 NULL 값 채우기
num_feats_to_use = list(clf_data3.select_dtypes(exclude=object).columns)
for feat in num_feats_to_use:
    imputer = Imputer(strategy='median') #median, mean, most_frequent
    clf_data3[feat] = imputer.fit_transform(clf_data3[feat].values.reshape(-1,1))

# 나눴던 두 테이블 Merge하기
merge_result = pd.merge(clf_data3, clf_data2, on = 'BeerID', how = 'left')

# StyleID와 나머지로 나누기
X = merge_result.iloc[:, 1:]     # 나머지
Y = merge_result.iloc[:, 0]     # StyleID

# Train/Test 나누기. TestSize는 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, test_size=.3)

# 무결성 확인. null값 남아있는지 확인.
X.info()

sanity_df = pd.DataFrame(X_train, columns = X.columns)
sanity_df.describe().T

# 스케일링 (Scaling)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#다시 무결성 체크
sanity_df = pd.DataFrame(X_train, columns = X.columns)
sanity_df.describe().T


# 알고리즘 import
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
clf1 = SVC()
clf1.fit(X_train, Y_train)
Y_pred1 = clf1.predict(X_test)
score1 = accuracy_score(Y_test, Y_pred1)
print('Accuracy: {}'.format(score1))

print(Y_train.head())
print(Y_test.head())


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
clf2 = RandomForestClassifier()
clf2.fit(X_train, Y_train)
Y_pred2 = clf2.predict(X_test)
score2 = metrics.accuracy_score(Y_test, Y_pred2)
score2 = accuracy_score(Y_test, Y_pred2)
print('Accuracy: {}'.format(score2))
print(score2)
cl_report = metrics.classification_report(Y_test, Y_pred2)
print("리포트 =\n", cl_report)

from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression()
clf3.fit(X_train, Y_train)
Y_pred3 = clf3.predict(X_test)
score3 = accuracy_score(Y_test, Y_pred3)
print('Accuracy: {}'.format(score3))


# 어떤 Feature가 영향력 있는지
feats_imp = pd.DataFrame(clf2.feature_importances_, index=X.columns, columns=['FeatureImportance'])
feats_imp = feats_imp.sort_values('FeatureImportance', ascending=False)

feats_imp.plot(kind='barh', figsize=(12,6), legend=False)
plt.title('Feature Importance from RandomForest Classifier')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()
