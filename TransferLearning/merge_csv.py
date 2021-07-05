import pandas as pd

feature_files = [
				'train_img_features_80009.csv',
				'train_img_features_20.csv',
				'train_img_features_30.csv',
				'train_img_features_remain.csv'
]

inter_files = [
				'train_img_features_inter_80009.csv',
				'train_img_features_inter_20.csv',
				'train_img_features_inter_30.csv',
				'train_img_features_inter_remain.csv'
]

merge_dfs = list()
for filename in feature_files:
	merge_dfs.append(pd.read_csv(filename))
features = pd.concat(merge_dfs, ignore_index=True)
print(features)
#features.to_csv('train_img_features.csv')

merge_dfs = list()
for filename in inter_files:
	merge_dfs.append(pd.read_csv(filename))
features_inter = pd.concat(merge_dfs, ignore_index=True)
print(features_inter)
features_inter.to_csv('train_img_features_inter.csv')