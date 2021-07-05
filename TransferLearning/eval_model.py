import lightgbm as lgbm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

model_name = 'run_7_3.model'
model = lgbm.Booster(model_file=model_name)
print(model_name)

def get_scores(valid_X, valid_y, disp=False):
	predict_probs = model.predict(valid_X)
	predict_classes = np.argmax(predict_probs, axis=-1)
	f_score = f1_score(valid_y, predict_classes, average='macro')
	if(disp):
		print('Accuracy : %.6f' % (accuracy_score(valid_y, predict_classes)))
		#print("Confusion Matrix:")
		#print(confusion_matrix(valid_y, predict_classes))
		#print()
		#print("Classification Report")
		print(classification_report(valid_y, predict_classes))
		print("F1 Score: ", f_score)
	return f_score

"""
df_train = pd.read_csv('valonly_metadata.csv')

print(df_train.info())
print(df_train.loc[:,['UUID', 'country', 'continent']])

train_feats = pd.read_csv('train_img_features_inter.csv')

print(train_feats.info())
print(train_feats)

# CSV columns
#['Unnamed: 0', 'binomial', 'country', 'continent', 'genus', 'family',
#       'UUID', 'source', 'subset', 'class_id', 'image_path']

# ['country', 'continent', 'file_path', 'UUID']

df_train_full = pd.merge(df_train, train_feats, how='inner', left_on='UUID', right_on='Unnamed: 0.1')
# print(df_train_full.columns)
# print(df_test_full.columns)
"""

#['Unnamed: 0', 'binomial', 'country', 'continent', 'genus', 'family', 'UUID', 'source', 'subset', 'class_id', 'image_path', 
#'Unnamed: 0.1', 'Unnamed: 0.1.1', 'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6']
#train = df_train_full.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis=1)
#test = df_test_full.drop(['image_name','patient_id'],axis=1)
#Drop the unwanted columns
#df_train_full.to_csv('eval_set.csv', index=False)
df_train_full = pd.read_csv('eval_set.csv')
#print(list(df_train_full.columns)[:20])
#print(df_train_full.info())
#print(df_train_full)

# Make train data
train = df_train_full.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'binomial', 'genus', 
								'family', 'UUID', 'source', 'subset', 'image_path'], axis=1)

features = [f for f in train.columns if f != 'class_id']

#print("TRAIN")
#print(train)

#Label Encode categorical features
train.country.fillna('unknown',inplace=True)
train.continent.fillna('unknown',inplace=True)
le_country = LabelEncoder()
le_continent = LabelEncoder()
train.country = le_country.fit_transform(train.country)
train.continent = le_continent.fit_transform(train.continent)

# Separate X and Y data
valid_X = train[features]
valid_y = train['class_id']
"""
scores_coun = list()
country_wise = df_train_full.groupby(['country'])
count = 0
for country, df in country_wise:
	count += 1
	train = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'binomial', 'genus', 
								'family', 'UUID', 'source', 'subset', 'image_path'], axis=1)
	train.country = le_country.transform(train.country)
	train.continent = le_continent.transform(train.continent)
	
	valid_X = train[features]
	valid_y = train['class_id']

	score = get_scores(valid_X, valid_y)
	scores_coun.append(score)
	print(country, score)

print("Macro Average (country): ", sum(scores_coun)/count, "across", count, "countries")

"""

get_scores(valid_X, valid_y, True)



# scp ~/Downloads/SnakeCLEF/ResNet/eval_model.py mirunap@mlrg-dl01.ssn.edu.in:~/SnakeCLEF2021/AugmentedModels/ResNet/