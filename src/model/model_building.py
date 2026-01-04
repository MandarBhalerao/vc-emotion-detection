import numpy as np
import pandas as pd
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier

import yaml
params=yaml.safe_load(open("params.yaml", "r"))['model_building']
n_estimators=params['n_estimators']
learning_rate=params['learning_rate']

train_data = pd.read_csv(os.path.join("data","features",'train_tfidf.csv'))

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:, -1].values

# Define and train the XGBoost model
clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
clf.fit(X_train, y_train)

# Save the trained model
pickle.dump(clf, open('models/model.pkl', 'wb'))

