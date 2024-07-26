import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('dataset/train_features.csv')
y_train = pd.read_csv('dataset/train_labels.csv')
X_test = pd.read_csv('dataset/test_features.csv')
y_test = pd.read_csv('dataset/test_labels.csv')

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

models = {
    "Perceptron": Perceptron(),
    "ADALINE": SGDClassifier(loss='perceptron', eta0=0.01, learning_rate='constant', max_iter=1000, tol=1e-3),
    "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear'),
    "Logistic Regression (L2)": LogisticRegression(penalty='l2'),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

for name, accuracy in accuracies.items():
    print(f'{name} Accuracy: {accuracy}')

accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
accuracy_df.to_csv('dataset/accuracies.csv', index=False)

print("Model accuracies saved to 'dataset/accuracies.csv'.")
