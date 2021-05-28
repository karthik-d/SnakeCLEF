import matplotlib.pyplot as plot 
import pandas as pd
import numpy as np

feat_imp_df = pd.read_csv("feat_imp.csv")
feature_imps = np.array(feat_imp_df.loc[:, ['importance']].values.tolist()).flatten()
feature_names = np.array(feat_imp_df.loc[:, ['feature']].values.tolist()).flatten()

required = 20 + 1

sorted_idx = np.argsort(feature_imps)
feature_imps = feature_imps[sorted_idx][-1:-required:-1]
feature_names = feature_names[sorted_idx][-1:-required:-1]

for idx,f_name in enumerate(feature_names):
	if f_name.startswith('feature'):
		feature_names[idx] = 'f{num}'.format(num=f_name.split('_')[-1])
	print(feature_names[idx])

for idx,val in enumerate(feature_imps[2:]):
	feature_imps[2+idx] *= 5


# Normalize between 0 and 100
feature_imps = 100*((feature_imps - feature_imps[-1]) / (feature_imps[0] - feature_imps[-1]))
plot.bar(feature_names, feature_imps)
plot.show()