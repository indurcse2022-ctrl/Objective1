import numpy as np
from sklearn.svm import SVC
import joblib

# dummy feature data
X = np.random.rand(100,1280)
y = np.random.randint(0,4,100)

svm = SVC(probability=True)

svm.fit(X,y)

joblib.dump(svm,"models/svm_model.pkl")

print("SVM model saved successfully")
