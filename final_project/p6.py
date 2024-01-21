import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

pas_class = []
pas_age = []
pas_survived = []
pas_gender = []
with open('TitanicPassengers.txt', 'r') as file:
    next(file)
    lines = file.readlines()
    for line in lines:
        val = line.split(',')
        pas_class.append(int(val[0]))
        pas_age.append(float(val[1]))
        pas_gender.append(1 if val[2] == 'M' else 0)
        pas_survived.append(int(val[3]))

X = np.array([pas_class, pas_age, pas_gender]).T
y = np.array(pas_survived)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k_values = list(range(1, 26, 2))
cv_scores = []
test_scores = []

k_values = list(range(1, 26, 2))
cv_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    test_scores.append(test_score)

    if k == 3:
        print("Results for k=3:")
        print("Cross Validation Accuracy:", round(scores.mean(), 3))
        print("Test Accuracy:", round(test_score, 3))
        C_M = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix for k=3:")
        print(C_M)
        print()

optimal_k = k_values[test_scores.index(max(test_scores))]
print("K for Maximum Accuracy is:", optimal_k)

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

C_M = confusion_matrix(y_test, y_pred)
print("Confusion Matrix for optimal k:")
print(C_M)

print("Predictions with maximum accuracy k:", optimal_k)
print("Cross Validation Accuracies is:", round(cv_scores[k_values.index(optimal_k)], 3))
print("Predicted Accuracies is:", round(test_scores[k_values.index(optimal_k)], 3))

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, label='n-fold cross validation')
plt.plot(k_values, test_scores, label='Real Prediction')
plt.xlabel('k values for KNN Regression')
plt.ylabel('Accuracy')
plt.xticks(range(1, 26, 2))
plt.title('Average Accuracy vs k (10 folds)')
plt.legend()
plt.savefig('Average Accuracy vs k (10 folds).png')