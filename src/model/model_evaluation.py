import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pickle
import json
import os

clf = pickle.load(open('models/model.pkl', 'rb'))

data_path = os.path.join("data","features")
test_data = pd.read_csv(os.path.join(data_path, 'test_bow.csv'))

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:, -1].values

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)


# print("Accuracy:", accuracy)
# print("Classification Report:\n", classification_rep)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "auc": auc
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)