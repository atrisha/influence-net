import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputRegressor

X = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
X_train = np.array(X)
Y = [[5.6,4.4],[6.6,5.5],[7.7,6.6],[8.8,9.9]]
Y_train = np.array(Y)

X_t = [[1.5,1.5,1.5,1.5]]
X_test = np.array(X_t)

clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train, Y_train)
clf_probs = clf.predict_proba(X_test)

print(clf_probs)