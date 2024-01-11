import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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

scaler = StandardScaler()

X = scaler.fit_transform(X)

weights = []
accuracies = []
sensitivities = []
specificities = []
ppv = []
auroc = []

for _ in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

print('Logistic Regression:')
print('Averages for all examples 1000 trials with k=0.5')
print('Mean weight of C1 = {}, Interval size = {}'.format(round(mean_weights[0], 3), round(upper_weights[0] - lower_weights[0], 3)))
print('Mean weight of age = {}, Interval size = {}'.format(round(mean_weights[1], 3), round(upper_weights[1] - lower_weights[1], 3)))
print('Mean accuracy = {}, Interval size = {}'.format(round(mean_accuracy, 3), round(upper_accuracy - lower_accuracy, 3)))
print('Mean sensitivity = {}, Interval size = {}'.format(round(mean_sensitivity, 3), round(upper_sensitivity - lower_sensitivity, 3)))
print('Mean specificity = {}, Interval size = {}'.format(round(mean_specificity, 3), round(upper_specificity - lower_specificity, 3)))
print('Mean pos. pred. val. = {}, Interval size = {}'.format(round(mean_ppv, 3), round(upper_ppv - lower_ppv, 3)))
print('Mean AUROC = {}, Interval size = {}'.format(round(mean_auroc, 3), round(upper_auroc - lower_auroc, 3)))

max_accuracies = np.max(accuracies)

plt.figure()
plt.hist(accuracies, bins=20, edgecolor='black')
plt.xlabel('Maximum Accuracies')
plt.ylabel('Numbers of Maximum Accuracies')
plt.title('Maximum Accuracies')
plt.savefig('Maximum Accuracies.png')

cor_counts = []

# Threshold values
k_values = np.linspace(0.4, 0.6, 10)

for k in k_values:
    y_pred = (clf.predict_proba(X_test)[:,1] >= k).astype(bool)

    cor_counts.append(np.sum(y_pred == y_test))

plt.figure()
plt.bar(k_values, cor_counts, width=0.01)
plt.xlabel('Threshold Values k')
plt.ylabel('Number of ks')
plt.title('Threshold values k for Maximum Accuracies')
plt.savefig('Threshold values k for Maximum Accuracies.png')

k_values = np.linspace(0.4, 0.6, 21)
accuracies = []

for k in k_values:
    y_pred = (clf.predict_proba(X_test)[:,1] >= k).astype(bool)

    accuracies.append(accuracy_score(y_test, y_pred))

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