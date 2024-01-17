import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


pas_class = []
pas_age = []
pas_gender = []
pas_survived = []
with open('TitanicPassengers.txt', 'r') as file:
    next(file)
    lines = file.readlines()
    for line in lines:
        val = line.split(',')
        pas_class.append(int(val[0]))
        pas_age.append(float(val[1]))
        if val[2] == 'M':
            pas_gender.append(1)
        else:
            pas_gender.append(0)
        pas_survived.append(int(val[3]))

pas_class_onehot = np.zeros((len(pas_class), max(pas_class)))
for i, c in enumerate(pas_class):
    pas_class_onehot[i, c-1] = 1

X = np.hstack((pas_class_onehot, np.array(pas_age).reshape(-1, 1), np.array(pas_gender).reshape(-1, 1)))
y = np.array(pas_survived)

weights = []
accuracies = []
sensitivities = []
specificities = []
ppv = []
auroc = []
max_accuracies = []
optimal_ks = []
optimal_k = 0.5
max_accuracy = 0
accuracies_for_k = {k: [] for k in np.linspace(0, 1, 100)}

scaler = MinMaxScaler()

for _ in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit on training set only.
    scaler.fit(X_train)

    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression()

    clf.fit(X_train, y_train)

    weights.append(clf.coef_[0])

    y_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracies.append(accuracy_score(y_test, y_pred))
    sensitivities.append(tp / (tp + fn))
    specificities.append(tn / (tn + fp))
    ppv.append(tp / (tp + fp))
    auroc.append(roc_auc_score(y_test, y_pred))
    
    optimal_k = 0.5
    max_accuracy = 0

    for k in np.linspace(0, 1, 100):
        y_pred = (clf.predict_proba(X_test)[:,1] >= k).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_for_k[k].append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_k = k

    optimal_ks.append(optimal_k)
    max_accuracies.append(max_accuracy)


weights = np.array(weights)

mean_weights = np.mean(weights, axis=0)
lower_weights, upper_weights = np.percentile(weights, [2.5, 97.5], axis=0)

mean_accuracy = np.mean(accuracies)
lower_accuracy, upper_accuracy = np.percentile(accuracies, [2.5, 97.5])

mean_sensitivity = np.mean(sensitivities)
lower_sensitivity, upper_sensitivity = np.percentile(sensitivities, [2.5, 97.5])

mean_specificity = np.mean(specificities)
lower_specificity, upper_specificity = np.percentile(specificities, [2.5, 97.5])

mean_ppv = np.mean(ppv)
lower_ppv, upper_ppv = np.percentile(ppv, [2.5, 97.5])

mean_auroc = np.mean(auroc)
lower_auroc, upper_auroc = np.percentile(auroc, [2.5, 97.5])

print('Logistic Regression with iScaling:')
print('Averages for all examples 1000 trials with k=0.5')
print('Mean weight of C1 = {}, 95% confidence interval = {}'.format(round(mean_weights[0], 3), round(upper_weights[0] - lower_weights[0], 3)))
print('Mean weight of C2 = {}, 95% confidence interval = {}'.format(round(mean_weights[1], 3), round(upper_weights[1] - lower_weights[1], 3)))
print('Mean weight of C3 = {}, 95% confidence interval = {}'.format(round(mean_weights[2], 3), round(upper_weights[2] - lower_weights[2], 3)))
print('Mean weight of age = {},  95% confidence interval = {}'.format(round(mean_weights[3], 3), round(upper_weights[3] - lower_weights[3], 3)))
print('Mean weight of Male Gender = {}, 95% CI = {}'.format(round(mean_weights[4], 3), round(upper_weights[4] - lower_weights[4], 3)))
print('Mean accuracy = {},  95% confidence interval = {}'.format(round(mean_accuracy, 3), round(upper_accuracy - lower_accuracy, 3)))
print('Mean sensitivity = {},  95% confidence interval = {}'.format(round(mean_sensitivity, 3), round(upper_sensitivity - lower_sensitivity, 3)))
print('Mean specificity = {},  95% confidence interval = {}'.format(round(mean_specificity, 3), round(upper_specificity - lower_specificity, 3)))
print('Mean pos. pred. val. = {},  95% confidence interval = {}'.format(round(mean_ppv, 3), round(upper_ppv - lower_ppv, 3)))
print('Mean AUROC = {},  95% confidence interval = {}'.format(round(mean_auroc, 3), round(upper_auroc - lower_auroc, 3)))

mean_optimal_k = np.mean(optimal_ks)
std_optimal_k = np.std(optimal_ks)
mean_max_accuracy = np.mean(max_accuracies)
std_max_accuracy = np.std(max_accuracies)

plt.figure(figsize=(10, 6))
plt.hist(optimal_ks, bins=20, range=(0.4, 0.6), edgecolor='black', label='k values for maximum accuracies\nMean ={:.2f} SD = {:.2f}'.format(mean_optimal_k, std_optimal_k))
plt.xlabel('Threshold Values k')
plt.ylabel('Number of ks')
plt.title('Threshold values k for Maximum Accuracies')
plt.savefig('Threshold values k for Maximum Accuracies.png')

max_accuracies = np.max(accuracies)

plt.figure()
plt.hist(accuracies, bins=20, edgecolor='black')
plt.xlabel('Maximum Accuracies')
plt.ylabel('Numbers of Maximum Accuracies')
plt.title('Maximum Accuracies')
plt.savefig('Maximum Accuracies.png')

k_values = np.linspace(0.4, 0.6, 20)
accuracies = np.zeros(len(k_values))

for _ in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    for i, k in enumerate(k_values):
        y_pred = (clf.predict_proba(X_test)[:,1] >= k).astype(bool)
        accuracies[i] += accuracy_score(y_test, y_pred)

accuracies /= 1000

max_accuracy_index = np.argmax(accuracies)
max_accuracy_k = k_values[max_accuracy_index]
max_accuracy = accuracies[max_accuracy_index]

plt.figure()
plt.plot(k_values, accuracies, label='Mean Accuracies')
plt.plot(max_accuracy_k, max_accuracy, 'ro', label='Max Mean Accuracies')
plt.annotate(f'({max_accuracy_k:.2f}, {max_accuracy:.2f})', (max_accuracy_k, max_accuracy), textcoords="offset points", xytext=(-10,-10), ha='center')
plt.xlabel('Threshold Values k')
plt.ylabel('Accuracy')
plt.title('Mean Accuracies for Different Threshold Values')
plt.legend()
plt.savefig('Mean Accuracies for Different Threshold Values.png')