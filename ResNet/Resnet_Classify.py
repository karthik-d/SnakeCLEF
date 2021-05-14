import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import gc

df_train = pd.read_csv('trainonly_metadata.csv')
df_test = pd.read_csv('valonly_metadata.csv')

train_feats = pd.read_csv('train_img_features.csv')
test_feats = pd.read_csv('test_img_features.csv')

# CSV columns
#['Unnamed: 0', 'binomial', 'country', 'continent', 'genus', 'family',
#       'UUID', 'source', 'subset', 'class_id', 'image_path'],

df_train_full = pd.merge(df_train, train_feats, how='inner', left_on='UUID', right_index=True)
df_test_full = pd.merge(df_test, test_feats, how='inner', left_on='UUID', right_index=True)

#train = df_train_full.drop(['image_name','patient_id','diagnosis','benign_malignant'],axis=1)
#test = df_test_full.drop(['image_name','patient_id'],axis=1)
#Drop the unwanted columns
train = df_train_full.drop(['Unnamed: 0', 'binomial', 'genus', 'family', 'UUID', 'source', 'subset', 'image_path'], axis=1)
test = df_test_full.drop(['Unnamed: 0', 'binomial', 'genus', 'family', 'UUID', 'source', 'subset', 'image_path'],axis=1)

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


# In[18]:


folds = StratifiedKFold(n_splits= 5, shuffle=True)
oof_preds = np.zeros(train.shape[0])    # Out of fold
sub_preds = np.zeros(test.shape[0])
feature_importance_df = pd.DataFrame()
features = [f for f in train.columns if f != 'class_id']
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train['class_id'])):
    train_X, train_y = train[features].iloc[train_idx], train['class_id'].iloc[train_idx]
    valid_X, valid_y = train[features].iloc[valid_idx], train['class_id'].iloc[valid_idx]
    clf = LGBMClassifier(
        #device='gpu',
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=8,
        colsample_bytree=0.5,
        num_leaves=50,
        random_state=23
    )
    print('*****Fold: {}*****'.format(n_fold))
    clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], 
            eval_metric= 'auc', verbose= 20, early_stopping_rounds= 20)

    oof_preds[valid_idx] = clf.predict_proba(valid_X, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del clf, train_X, train_y, valid_X, valid_y
    gc.collect()


# In[19]:


submission = pd.DataFrame({
    "UUID": df_test.UUID, 
    "prediction": sub_preds
})
submission.to_csv('submission.csv', index=False)

