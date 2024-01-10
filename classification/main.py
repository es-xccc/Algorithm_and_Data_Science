import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np
import random
# import chap2408


rings = []
with open('abalone.txt', 'r') as f:
    next(f)
    for line in f:
        row = line.strip().split(',')
        rings.append(int(row[-1]))

test_size = int(len(rings) * 0.2)
test_rings = random.sample(rings, test_size)

mean = statistics.mean(rings)
std_dev = statistics.stdev(rings)
test_mean = statistics.mean(test_rings)
test_std_dev = statistics.stdev(test_rings)

plt.figure(figsize=(10, 6))
plt.hist(rings, bins=range(min(rings), max(rings) + 1), alpha=0.7, edgecolor='black')
plt.title('All Abalone Ring Sizes (Age) Distribution')
plt.xlabel('Rings Sizes')
plt.ylabel('Number of Abalones')
plt.text(0.95, 0.95, f'Mean = {mean:.2f}\nSD = {std_dev:.2f}', ha='right', va='top', transform=plt.gca().transAxes)
plt.savefig('All Abalone Ring Sizes (Age) Distribution.png')

plt.figure(figsize=(10, 6))
plt.hist(test_rings, bins=range(min(test_rings), max(test_rings) + 1), alpha=0.7, edgecolor='black')
plt.title('Test Set Ring Sizes (Age) Distribution')
plt.xlabel('Rings Sizes')
plt.ylabel('Number of Abalones')
plt.text(0.95, 0.95, f'Mean = {test_mean:.2f}\nSD = {test_std_dev:.2f}', ha='right', va='top', transform=plt.gca().transAxes)
plt.savefig('Test Set Ring Sizes (Age) Distribution.png')



features = []
targets = []
with open('abalone.txt', 'r') as f:
    next(f)
    for line in f:
        row = line.strip().split(',')
        features.append([float(x) for x in row[1:-1]])
        targets.append(int(row[-1]))

features = np.array(features)
targets = np.array(targets)

test_k = [3, 5, 7, 9, 11]
predictions = {}

features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, random_state=42)

for k in test_k:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(features_train, targets_train)
    predictions_train = knn.predict(features_train)
    rSquare_train = r2_score(targets_train, predictions_train)
    rmse_train = sqrt(mean_squared_error(targets_train, predictions_train))
    print(f"Training with Whole Examples Evaluation with k={k}")
    print(f"Coefficient of Determination: rSquare(R²): {rSquare_train:.4f}")
    print(f"Root Mean Square Deviation Rmsd: {rmse_train:.4f}")

print('-' * 50)
for k in test_k:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(features_train, targets_train)
    predictions_test = knn.predict(features_test)
    rSquare_test = r2_score(targets_test, predictions_test)
    rmse_test = sqrt(mean_squared_error(targets_test, predictions_test))
    print(f"After Trained Testing Using Test Set with k={k}")
    print(f"Coefficient of Determination: rSquare(R²): {rSquare_test:.4f}")
    print(f"Root Mean Square Deviation Rmsd: {rmse_test:.4f}")

# import chap2408
import pandas as pd
ring_sizes = targets_test
absolute_errors = np.abs(targets_test - predictions_test)
percentage_errors = absolute_errors / targets_test * 100

df = pd.DataFrame({
    'ring_sizes': ring_sizes,
    'absolute_errors': absolute_errors,
    'percentage_errors': percentage_errors
})

average_errors = df.groupby('ring_sizes').mean()

plt.figure(figsize=(10, 5))

plt.plot(average_errors.index, average_errors['absolute_errors'], marker='o', label='Average Absolute Error')

plt.plot(average_errors.index, average_errors['percentage_errors'], marker='o', label='Average Percentage Error')

plt.title('Predict Rings Absolute and Percentage Error for Each Ring Size')
plt.xlabel('Ring sizes')
plt.ylabel('Inches and Percent')
plt.legend()

plt.tight_layout()
plt.savefig('Predict Rings Absolute and Percentage Error for Each Ring Size.png')


import chap2408
k_values = [3, 5, 7, 9, 11]

for k in k_values:
    test_rings = random.sample(rings, test_size)

    test_mean = statistics.mean(test_rings)
    test_std_dev = statistics.stdev(test_rings)

    plt.figure(figsize=(10, 6))
    plt.hist(test_rings, bins=range(min(test_rings), max(test_rings) + 1), alpha=0.7, edgecolor='black')
    plt.title(f'Predicted Rings (Ages) Absolute Errors Distribution for k={k}')
    plt.xlabel('Predicted Ring Sizes Absolute Errors')
    plt.ylabel('Number of Abalones')
    plt.tight_layout()
    plt.savefig(f'Predicted Rings Absolute Errors Distribution for k={k}.png')


for k in k_values:
    test_rings = random.sample(rings, test_size)

    test_mean = statistics.mean(test_rings)
    test_std_dev = statistics.stdev(test_rings)

    # 計算比例誤差
    percentage_errors = [(ring - test_mean) / test_mean * 100 for ring in test_rings]

    plt.figure(figsize=(10, 6))
    plt.hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Predicted Rings (Ages) Percentage Errors Distribution for k={k}')
    plt.xlabel('Predicted Ring Sizes Percentage Errors')
    plt.ylabel('Number of Abalones')
    plt.tight_layout()
    plt.savefig(f'Predicted Rings Percentage Errors Distribution for k={k}.png')