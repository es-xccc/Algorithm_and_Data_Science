import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

male_class = []
male_age = []
male_survived = []
female_class = []
female_age = []
female_survived = []
with open('TitanicPassengers.txt', 'r') as file:
    next(file)
    lines = file.readlines()
    for line in lines:
        val = line.split(',')
        if(val[2] == 'M'):
            male_class.append(val[0])
            male_age.append(val[1])
            male_survived.append(val[3])
        else:
            female_class.append(val[0])
            female_age.append(val[1])
            female_survived.append(val[3])

male_class = np.array(male_class, dtype=int)
male_age = np.array(male_age, dtype=float)
male_survived = np.array(male_survived, dtype=int)

X_male = np.vstack((male_class, male_age)).T
y_male = male_survived

X_train, X_test, y_train, y_test = train_test_split(X_male, y_male, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
ppv = precision_score(y_test, y_pred)

print('Try to predict male and female separately and combined with k=3:')
print('For Male:')
print(f"Confusion Matrix is: \n{cm}")
print(f"Accuracy = {accuracy}")
print(f"Sensitivity = {sensitivity}")
print(f"Specificity = {specificity}")
print(f"Pos. Pred. Val. = {ppv}")

female_class = np.array(female_class, dtype=int)
female_age = np.array(female_age, dtype=float)
female_survived = np.array(female_survived, dtype=int)

X_female = np.vstack((female_class, female_age)).T
y_female = female_survived
X_train, X_test, y_train, y_test = train_test_split(X_female, y_female, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
ppv = precision_score(y_test, y_pred)

print('Try to predict male and female separately and combined with k=3:')
print('For Female:')
print(f"Confusion Matrix is: \n{cm}")
print(f"Accuracy = {accuracy}")
print(f"Sensitivity = {sensitivity}")
print(f"Specificity = {specificity}")
print(f"Pos. Pred. Val. = {ppv}")

knn.fit(X_female, y_female)
y_pred_female = knn.predict(X_test)

cm_female = confusion_matrix(y_test, y_pred_female)
accuracy_female = accuracy_score(y_test, y_pred_female)
sensitivity_female = recall_score(y_test, y_pred_female)
specificity_female = cm_female[0, 0] / (cm_female[0, 0] + cm_female[0, 1])
ppv_female = precision_score(y_test, y_pred_female)

knn.fit(X_male, y_male)
y_pred_male = knn.predict(X_test)

cm_male = confusion_matrix(y_test, y_pred_male)
accuracy_male = accuracy_score(y_test, y_pred_male)
sensitivity_male = recall_score(y_test, y_pred_male)
specificity_male = cm_male[0, 0] / (cm_male[0, 0] + cm_male[0, 1])
ppv_male = precision_score(y_test, y_pred_male)

cm_combined = cm_female + cm_male
accuracy_combined = (accuracy_female + accuracy_male) / 2
sensitivity_combined = (sensitivity_female + sensitivity_male) / 2
specificity_combined = (specificity_female + specificity_male) / 2
ppv_combined = (ppv_female + ppv_male) / 2

print('For Combined:')
print(f"Confusion Matrix is: \n{cm_combined}")
print(f"Accuracy = {accuracy_combined}")
print(f"Sensitivity = {sensitivity_combined}")
print(f"Specificity = {specificity_combined}")
print(f"Pos. Pred. Val. = {ppv_combined}")