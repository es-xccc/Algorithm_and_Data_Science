import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

pas_class = []
pas_age = []
pas_survived = []
with open('TitanicPassengers.txt', 'r') as file:
    next(file)
    lines = file.readlines()
    for line in lines:
        val = line.split(',')
        pas_class.append(int(val[0]))
        pas_age.append(float(val[1]))
        pas_survived.append(int(val[3]))


X = np.array([pas_class, pas_age]).T
y = np.array(pas_survived)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for k=3:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
ppv = precision_score(y_test, y_pred)

print(f"Accuracy = {accuracy:.3f}\nSensitivity = {sensitivity:.3f}\nSpecificity = {specificity:.3f}\nPos. Pred. Val. = {ppv:.3f}")

k_values = list(range(1, 31))
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[cv_scores.index(max(cv_scores))]
print("K for Maximum Accuracy is:", optimal_k)

knn = KNeighborsClassifier(n_neighbors=optimal_k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for optimal k:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
ppv = precision_score(y_test, y_pred)

print(f"Accuracy = {accuracy:.3f}\nSensitivity = {sensitivity:.3f}\nSpecificity = {specificity:.3f}\nPos. Pred. Val. = {ppv:.3f}")

real_pred_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    real_pred_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='b', label='n-fold cross validation')
plt.plot(k_values, real_pred_scores, marker='s', linestyle='--', color='r', label='Real Prediction')
plt.xlabel('k values for KNN Regression')
plt.ylabel('Accuracy')
plt.title('Average Accuracy vs k (10 folds)')
plt.legend()
plt.savefig('Average Accuracy vs k (10 folds).png')