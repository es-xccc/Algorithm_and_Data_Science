import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

with open('TestScorerResult.txt', 'r') as file:
    lines = file.readlines()

scores = [float(score) for line in lines for score in line.split(',')]

mean = np.mean(scores)
std_dev = np.std(scores)

plt.figure()
plt.hist(scores, bins=50, edgecolor='black')
plt.title('2012 Test-Scores-All-12500')
plt.xlabel('Test Score')
plt.ylabel('Number of Students')
plt.text(0.2, 0.8, f'Mean: {mean:.2f}\nSD: {std_dev:.2f}', transform=plt.gca().transAxes, fontsize=14)
plt.savefig('2012 Test-Scores-All-12500.png')

plt.figure()
x = np.linspace(min(scores), max(scores), 100)
y = norm.pdf(x, mean, std_dev)
plt.plot(x, y)
plt.title('Normal Distribution')
y_mean = norm.pdf(mean, mean, std_dev)
plt.plot([mean, mean], [0, y_mean], color='black')
y_std_dev1 = norm.pdf(mean - 1.96*std_dev, mean, std_dev)
y_std_dev2 = norm.pdf(mean + 1.96*std_dev, mean, std_dev)
plt.plot([mean - 1.96*std_dev, mean - 1.96*std_dev], [0, y_std_dev1], color='black', linestyle='dashed')
plt.plot([mean + 1.96*std_dev, mean + 1.96*std_dev], [0, y_std_dev2], color='black', linestyle='dashed')
plt.savefig('Normal Distribution.png')

num_scores_above_60 = sum(score >= 60 for score in scores)
percentage_scores_above_60 = round(num_scores_above_60 / 125)
print(f'Real number of scores above 60 is: {num_scores_above_60}, around {percentage_scores_above_60}%')

num_scores_within_range = sum(mean - 1.96*std_dev <= score <= mean + 1.96*std_dev for score in scores)
print(f'Precise 95% number of scores between mu-1.96*sigma and mu+1.96*sigma is: {num_scores_within_range}')
# Calculate the number of scores below mean - 1.96 standard deviations
num_scores_below_range = sum(score < mean - 1.96*std_dev for score in scores)
print(f'Number of scores below mu-1.96*sigma({round(mean - 1.96*std_dev, 2)}) is: {num_scores_below_range}')

# Calculate the number of scores above mean + 1.96 standard deviations
num_scores_above_range = sum(score > mean + 1.96*std_dev for score in scores)
print(f'Number of scores above mu+1.96*sigma({round(mean + 1.96*std_dev, 2)}) is: {num_scores_above_range}')

import random

sample_sizes = [50, 200, 500]
sample_means = []
sample_std_devs = []
standard_errors = []
estimated_standard_errors = []

for size in sample_sizes:
    sample = random.sample(scores, size)
    sample_mean = np.mean(sample)
    sample_std_dev = np.std(sample)
    sample_means.append(sample_mean)
    sample_std_devs.append(sample_std_dev)
    standard_errors.append(std_dev / np.sqrt(size))
    estimated_standard_errors.append(sample_std_dev / np.sqrt(size))

offset = 10
standard_error_percentages = [error / mean * 100 * 2 for error in standard_errors]
estimated_standard_error_percentages = [error / mean * 100 * 2 for error in estimated_standard_errors]
plt.figure()
x_values = [50, 200, 500]
y_values = [error / sample_mean * 100 + sample_mean for error, sample_mean in zip(standard_errors, sample_means)]
for x, y, percentage in zip(x_values, y_values, standard_error_percentages):
    plt.text(x, y, str(round(percentage, 2)) + '%', color='orange')
y_values = [- error / sample_mean * 100 + sample_mean for error, sample_mean in zip(estimated_standard_errors, sample_means)]
for x, y, percentage in zip(x_values, y_values, estimated_standard_error_percentages):
    plt.text(x, y, str(round(percentage, 2)) + '%', color='blue')
plt.errorbar([size - offset for size in sample_sizes], sample_means, yerr=standard_errors, fmt='o', label='95% SE confidence interval')
plt.errorbar(sample_sizes, sample_means, yerr=estimated_standard_errors, fmt='o-', label='95% Estimated SE confidence interval')
plt.axhline(y=mean, color='blue', linestyle='--', label='True mean')
plt.title('Estimated of Mean Scores')
plt.xlabel('Sample Size')
plt.ylabel('Scores')
plt.xticks(np.arange(0, 700, 100))
plt.yticks(np.arange(65, 75, 1))
plt.legend()
plt.savefig('Estimated of Mean Scores.png')

