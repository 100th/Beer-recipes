# https://www.kaggle.com/jtrofe/beer-recipes/
# main_top10.py
import os
import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics
sns.set(style="whitegrid")


# 저장 폴더 지정
folder = 'C:/Users/paramount/Desktop/Github/Beer-recipes/result'

# 데이터 head 보기
beer_recipe = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Beer-recipes/recipeData_cleansing2.csv', index_col='BeerID', encoding='latin1')
beer_recipe.columns.values[2] = "Size"


# 그룹화
style_cnt = beer_recipe.loc[:,['StyleID']]
style_cnt['Count'] = 1
style_cnt_grp = style_cnt.loc[:,['StyleID','Count']].groupby('StyleID').sum()
style_cnt_grp = style_cnt_grp.sort_values('Count', ascending=False)
style_cnt_grp.reset_index(inplace=True)
#print(style_cnt_grp)


# SugarScale에 뭐가 몇 개 들어있는지 Count
# All Grain, BIAB, Partial Mash, extract --------------> SugarScale로 바꿈
beer_recipe.loc[beer_recipe.SugarScale == 'All Grain', 'SugarScale'] = None
beer_recipe.loc[beer_recipe.SugarScale == 'BIAB', 'SugarScale'] = None
beer_recipe.loc[beer_recipe.SugarScale == 'Partial Mash', 'SugarScale'] = None
beer_recipe.loc[beer_recipe.SugarScale == 'extract', 'SugarScale'] = None
# ax = sns.countplot(x='SugarScale', data=beer_recipe)
# ax.set(title='Frequency table of possible values in SugarScale')
# sns.despine(left=True, bottom=True)
# print('SugarScale has {} null values'.format(beer_recipe.SugarScale.isnull().sum()))


# BrewMethod에 뭐가 몇 개 들어있는지 Count
# ax = sns.countplot(x='BrewMethod', data=beer_recipe)
# ax.set(title='Frequency table of possible values in BrewMethod')
# sns.despine(left=True, bottom=True)
# print('BrewMethod has {} null values'.format(beer_recipe.BrewMethod.isnull().sum()))


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
# num_feats_list = ['Size', 'OG', 'OG_sg', 'FG', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'BrewMethod']
# beer_recipe.loc[:, num_feats_list].describe().T


# 지표값 범위로 분류하기
vvlow_scale_feats = ['OG_sg', 'FG_sg']
vlow_scale_feats = ['BoilGravity_sg']
low_scale_feats = ['ABV', 'BrewMethod']
mid_scale_feats = ['Color', 'BoilTime', 'Efficiency', 'IBU']
high_scale_feats = ['Size', 'BoilSize']

# vvlow_scale_feats
beer_recipe.loc[beer_recipe.OG_sg > 1.057 + 1.5 * (1.067 - 1.050), 'OG_sg'] = None
beer_recipe.loc[beer_recipe.FG_sg > 1.013 + 1.5 * (1.016 - 1.011), 'FG_sg'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, vvlow_scale_feats], orient='h')
ax.set(title='Boxplots of very very low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

# vlow_scale_feats
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, vlow_scale_feats], orient='h')
ax.set(title='Boxplots of very low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

# low_scale_feats
beer_recipe.loc[beer_recipe.ABV > 5.8 + 1.5 * (6.83 - 5.09), 'ABV'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, low_scale_feats], orient='h')
ax.set(title='Boxplots of low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

#mid_scale_feats
beer_recipe.loc[beer_recipe.IBU > 200, 'IBU'] = None
beer_recipe.loc[beer_recipe.Color > 80, 'Color'] = None
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, mid_scale_feats], orient='h')
ax.set(title='Boxplots of medium scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)

#high_scale_feats
f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, high_scale_feats], orient='h')
ax.set(title='Boxplots of high scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)


# 원 그래프 그리기
top10_style = list(style_cnt_grp['StyleID'][:10].values)
style_cnt_other = style_cnt_grp.loc[:, ['StyleID','Count']]
style_cnt_other.StyleID = style_cnt_grp.StyleID.apply(lambda x: x if x in top10_style else 'Other')
style_cnt_other = style_cnt_other.groupby('StyleID').sum()
style_cnt_other['Ratio'] = style_cnt_other.Count.apply(lambda x: x/float(len(beer_recipe)))
style_cnt_other = style_cnt_other.sort_values('Count', ascending=False)
f, ax = plt.subplots(figsize=(8, 8))
plt.pie(x=style_cnt_other['Ratio'], labels=list(style_cnt_other.index), startangle = 180, autopct='%1.1f%%', pctdistance= .9)
plt.title('Ratio of styles across dataset')
plt.show()


# 상관관계 그림. 우리가 관심있는 것만 선택해서
# pairplot_df = beer_recipe.loc[:, ['Efficiency', 'MashThickness']]    #'StyleID', 'Size', 'OG_sg', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'BrewMethod'
# len(pairplot_df)
# pairplot_df2 = pairplot_df.dropna()     # dropna 하면 69656개 -> 41900개
# len(pairplot_df2)
# sns.set(style="dark")
# sns.pairplot(data=pairplot_df2)
# plt.show()


# 사용할 Feature 설정
features_list= ['StyleID', 'ABV','IBU','Color', 'OG_sg','FG_sg', 'Size',
                'BoilSize', 'Efficiency', 'BoilGravity_sg', 'BrewMethod']
clf_data = beer_recipe.loc[:, features_list]


# 결측치 제거 두 가지 방법
# 1. 하나라도 Null이 있으면 그 행 제거.     2. 평균으로 채워넣기
include_object_list = []
clf_data2 = beer_recipe.loc[:, include_object_list]
exclude_object_list = ['StyleID', 'ABV', 'IBU', 'Color', 'OG_sg','FG_sg', 'Size', 'BoilSize', 'Efficiency', 'BoilGravity_sg', 'BrewMethod']
clf_data3 = beer_recipe.loc[:, exclude_object_list]
clf_data2 = clf_data2.dropna()     # clf_data2는 문자라서 dropna() 썼다.
clf_data3 = clf_data3.fillna(clf_data3.mean())   # clf_data3는 숫자라서 평균으로 채웠다.

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
# X.info()
sanity_df = pd.DataFrame(X_train, columns = X.columns)
remove_outlier = sanity_df.describe().T
remove_outlier_path = os.path.join(folder, '1. remove_outlier_top10.csv')
remove_outlier.to_csv(remove_outlier_path)


# 스케일링 (Scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sanity_df = pd.DataFrame(X_train, columns = X.columns)
scaling = sanity_df.describe().T
scaling_path = os.path.join(folder, '2. scaling_top10.csv')
scaling.to_csv(scaling_path)


# 1.서포트 벡터 머신
# from sklearn.svm import SVC
# clf1 = SVC()
# clf1.fit(X_train, Y_train)
# Y_pred1 = clf1.predict(X_test)
# score1 = accuracy_score(Y_test, Y_pred1)
# print('Accuracy: {}'.format(score1))


# 2.랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier()
clf2.fit(X_train, Y_train)
Y_pred2 = clf2.predict(X_test)
score2 = accuracy_score(Y_test, Y_pred2)
print('Accuracy: {}'.format(score2))
print(score2)
cl_report = metrics.classification_report(Y_test, Y_pred2)
print(cl_report)


# 3.로지스틱 회귀
# from sklearn.linear_model import LogisticRegression
# clf3 = LogisticRegression()
# clf3.fit(X_train, Y_train)
# Y_pred3 = clf3.predict(X_test)
# score3 = accuracy_score(Y_test, Y_pred3)
# print('Accuracy: {}'.format(score3))


# 어떤 Feature가 영향력 있는지
feats_imp = pd.DataFrame(clf2.feature_importances_, index=X.columns, columns=['FeatureImportance'])
feats_imp = feats_imp.sort_values('FeatureImportance', ascending=False)
feats_imp.plot(kind='barh', figsize=(12,6), legend=False)
plt.title('Feature Importance from RandomForest Classifier')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()
