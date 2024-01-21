import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

X_male = X[X[:, 2] == 1]
y_male = y[X[:, 2] == 1]
X_female = X[X[:, 2] == 0]
y_female = y[X[:, 2] == 0]

k = 3

X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2)
knn_male = KNeighborsClassifier(n_neighbors=k)
knn_male.fit(X_train_male, y_train_male)
y_pred_male = knn_male.predict(X_test_male)

C_M_male = confusion_matrix(y_test_male, y_pred_male)
print("For male:")
print(C_M_male)

accuracy_male = round((C_M_male[0][0] + C_M_male[1][1]) / np.sum(C_M_male), 3)
sensitivity_male = round(C_M_male[1][1] / (C_M_male[1][1] + C_M_male[1][0]), 3)
specificity_male = round(C_M_male[0][0] / (C_M_male[0][0] + C_M_male[0][1]), 3)
ppv_male = round(C_M_male[1][1] / (C_M_male[1][1] + C_M_male[0][1]), 3)

print("Accuracy:", accuracy_male)
print("Sensitivity:", sensitivity_male)
print("Specificity:", specificity_male)
print("Pos. Pred. Val.:", ppv_male)


X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.2)
knn_female = KNeighborsClassifier(n_neighbors=k)
knn_female.fit(X_train_female, y_train_female)
y_pred_female = knn_female.predict(X_test_female)

C_M_female = confusion_matrix(y_test_female, y_pred_female)
print("For female:")
print(C_M_female)

accuracy_female = round((C_M_female[0][0] + C_M_female[1][1]) / np.sum(C_M_female), 3)
sensitivity_female = round(C_M_female[1][1] / (C_M_female[1][1] + C_M_female[1][0]), 3)
specificity_female = round(C_M_female[0][0] / (C_M_female[0][0] + C_M_female[0][1]), 3)
ppv_female = round(C_M_female[1][1] / (C_M_female[1][1] + C_M_female[0][1]), 3)

print("Accuracy:", accuracy_female)
print("Sensitivity:", sensitivity_female)
print("Specificity:", specificity_female)
print("Pos. Pred. Val.:", ppv_female)

truePos_combined = C_M_male[1][1] + C_M_female[1][1]
falsePos_combined = C_M_male[0][1] + C_M_female[0][1]
trueNeg_combined = C_M_male[0][0] + C_M_female[0][0]
falseNeg_combined = C_M_male[1][0] + C_M_female[1][0]

print("Combined Predictions Statistics:")
print('TP,FP,TN,FN = ', truePos_combined, falsePos_combined, trueNeg_combined, falseNeg_combined)

accuracy_combined = round((trueNeg_combined + truePos_combined) / (truePos_combined + falsePos_combined + trueNeg_combined + falseNeg_combined), 3)
sensitivity_combined = round(truePos_combined / (truePos_combined + falseNeg_combined), 3)
specificity_combined = round(trueNeg_combined / (trueNeg_combined + falsePos_combined), 3)
ppv_combined = round(truePos_combined / (truePos_combined + falsePos_combined), 3)

print("Accuracy:", accuracy_combined)
print("Sensitivity:", sensitivity_combined)
print("Specificity:", specificity_combined)
print("Pos. Pred. Val.:", ppv_combined)