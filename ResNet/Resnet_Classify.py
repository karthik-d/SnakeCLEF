import pandas as pd
from lightgbm import LGBMClassifier, print_evaluation
#from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
import gc
import numpy as np
import os

RESULT_PATH = os.path.join(os.getcwd(), 'Results')

# df_train = pd.read_csv('trainonly_metadata.csv')
# df_test = pd.read_csv('valonly_metadata.csv')
df_train = pd.read_csv('train_metadata.csv')
df_test = pd.read_csv('test_metadata.csv')

print(df_train.info())
print(df_train.loc[:,['UUID', 'country', 'continent']])

train_feats = pd.read_csv('train_img_features_inter.csv')
test_feats = pd.read_csv('test_img_features_inter.csv')

print(train_feats.info())
print(train_feats)

# CSV columns
#['Unnamed: 0', 'binomial', 'country', 'continent', 'genus', 'family',
#       'UUID', 'source', 'subset', 'class_id', 'image_path']

# ['country', 'continent', 'file_path', 'UUID']

df_train_full = pd.merge(df_train, train_feats, how='inner', left_on='UUID', right_on='Unnamed: 0.1')
df_test_full = pd.merge(df_test, test_feats, how='inner', left_on='UUID', right_on='Unnamed: 0')
# print(df_train_full.columns)
# print(df_test_full.columns)

#train = df_train_full.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis=1)
#test = df_test_full.drop(['image_name','patient_id'],axis=1)
#Drop the unwanted columns
train = df_train_full.drop(['Unnamed: 0', 'Unnamed: 0.1', 'binomial', 'genus', 'family', 'UUID', 'source', 'subset', 'image_path'], axis=1)
test = df_test_full.drop(['Unnamed: 0', 'UUID', 'file_path'], axis=1)

print("TRAIN")
print(train)
print("TEST")
print(test)

#Label Encode categorical features
train.country.fillna('unknown',inplace=True)
test.country.fillna('unknown',inplace=True)
train.continent.fillna('unknown',inplace=True)
test.continent.fillna('unknown',inplace=True)
le_country = LabelEncoder()
le_continent = LabelEncoder()
train.country = le_country.fit_transform(train.country)
test.country = le_country.transform(test.country)
train.continent = le_continent.fit_transform(train.continent)
test.continent = le_continent.transform(test.continent)

#print(len(train.class_id.unique()))


# In[18]:


folds = StratifiedKFold(n_splits= 5, shuffle=True)
oof_preds = np.zeros(train.shape[0])    # Out of fold
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
features = [f for f in train.columns if f != 'class_id']
print(features)


"""
leaves -> 2^depth
depth -> 8
iter -> 50
"""
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train['class_id'])):
    train_X, train_y = train[features].iloc[train_idx], train['class_id'].iloc[train_idx]
    valid_X, valid_y = train[features].iloc[valid_idx], train['class_id'].iloc[valid_idx]
    # print(train_X)
    # print(valid_X)
    # print(train_y)
    # print(valid_y)
    clf = LGBMClassifier(
        device='cpu',
		objective='multiclass',
		num_classes=772,
		num_iterations=5,
		learning_rate=0.006,
		num_leaves=128,
		random_state=23,
		max_depth=-1,
		colsample_bytree=0.8
    )
    """
		n_estimators=1000,
        learning_rate=0.001,
        max_depth=8,
        colsample_bytree=0.5,
        num_leaves=50,
        random_state=23
    """
    saved_model = 'run_7_4.model'
    for i in range(5, 20):	
        print('*****Fold: {}*****'.format(n_fold))
        clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], 
				eval_metric= 'multi_logloss', verbose=1, init_model=saved_model)
        oof_preds[valid_idx] = clf.predict(valid_X, num_iteration=clf.best_iteration_)
        sub_preds = clf.predict(test[features], num_iteration=clf.best_iteration_)
        print('Fold %2d Accuracy : %.6f' % (n_fold + 1, accuracy_score(valid_y, oof_preds[valid_idx])))
        saved_model = "run_7_{0}.model".format(i)
        clf.booster_.save_model(saved_model)
        submission = pd.DataFrame({
		"UUID": df_test.UUID, 
		"prediction": sub_preds
        })
        submission.to_csv(os.path.join(RESULT_PATH, 'submission_{fold_num}.csv'.format(fold_num=i)), index=False)


    oof_preds[valid_idx] = clf.predict(valid_X, num_iteration=clf.best_iteration_)
    sub_preds = clf.predict(test[features], num_iteration=clf.best_iteration_)#[:, 1] / folds.n_splits
    #sub_preds_prob = clf.predict_proba(test[features], num_iteration=clf.best_iteration_)
    print(sub_preds)
    #print(sub_preds_prob)

    print("Writing to dataframe...")
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    feature_importance_df.to_csv(os.path.join(RESULT_PATH, 'feature_importance.csv'))
    
    del clf, train_X, train_y, valid_X, valid_y
    gc.collect()
    
    submission = pd.DataFrame({
		"UUID": df_test.UUID, 
		"prediction": sub_preds
    })
    submission.to_csv(os.path.join(RESULT_PATH, 'submission_{fold_num}.csv'.format(fold_num=n_fold)), index=False)
    """
    class_probs = pd.DataFrame(zip(df_test.UUID, sub_preds_prob))
    class_probs.to_csv(os.path.join(RESULT_PATH, 'probs_{fold_num}.csv'.format(fold_num=n_fold)), index=False)
    """

# In[19]:
'''
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train['class_id'])):
    train_X, train_y = train[features].iloc[train_idx], train['class_id'].iloc[train_idx]
    valid_X, valid_y = train[features].iloc[valid_idx], train['class_id'].iloc[valid_idx]
    clf = XGBClassifier(use_label_encoder=False)
    print('*****Fold: {}*****'.format(n_fold))
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], 
            eval_metric= 'mlogloss', verbose=True, early_stopping_rounds=10, tree_method='gpu_hist')

    oof_preds[valid_idx] = clf.predict_proba(valid_X, num_iteration=clf.best_iteration)[:, 1]
    sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration)[:, 1] / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_X, train_y, valid_X, valid_y
    gc.collect()
'''
"""
submission = pd.DataFrame({
    "UUID": df_test.UUID, 
    "prediction": sub_preds
})
submission.to_csv('submission.csv', index=False)
"""
# CUDA Library paths
# /usr/local/cuda-11.2/lib64/libOpenCL.so
# /usr/local/cuda-11.2/include/

# scp ~/Downloads/SnakeCLEF/ResNet/Resnet_Classify.py mirunap@mlrg-dl01.ssn.edu.in:~/SnakeCLEF2021/AugmentedModels/ResNet/