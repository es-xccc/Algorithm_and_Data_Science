import matplotlib.pyplot as plt
import numpy as np

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

male_age = [int(float(age)) for age in male_age]
male_survived_age = [male_age[i] for i in range(len(male_age)) if male_survived[i] == '1']

bins = 20

male_age_mean = np.mean(male_age)
male_age_std = np.std(male_age)
male_survived_age_mean = np.mean(male_survived_age)
male_survived_age_std = np.std(male_survived_age)
all_male_label = 'All Male Passengers\nMean = {:.2f} SD = {:.2f}'.format(male_age_mean, male_age_std)
survived_male_label = 'Survived Male Passengers\nMean = {:.2f} SD = {:.2f}'.format(male_survived_age_mean, male_survived_age_std)

plt.figure()
plt.hist(male_age, bins=bins, label=all_male_label, edgecolor='black')
plt.hist(male_survived_age, bins=bins, label=survived_male_label, edgecolor='black')
plt.xlabel('Male Ages')
plt.ylabel('Number of Male Passengers')
plt.title('Male Passengers and Survived')
plt.legend(loc='upper right')
plt.savefig('Male Passengers and Survived.png')

female_age = [int(float(age)) for age in female_age]
female_survived_age = [female_age[i] for i in range(len(female_age)) if female_survived[i] == '1']

bins = 20

female_age_mean = np.mean(female_age)
female_age_std = np.std(female_age)
female_survived_age_mean = np.mean(female_survived_age)
female_survived_age_std = np.std(female_survived_age)

all_female_label = 'All Female Passengers\nMean = {:.2f} SD = {:.2f}'.format(female_age_mean, female_age_std)
survived_female_label = 'Survived Female Passengers\nMean = {:.2f} SD = {:.2f}'.format(female_survived_age_mean, female_survived_age_std)

plt.figure()
plt.hist(female_age, bins=bins, label=all_female_label, edgecolor='black')
plt.hist(female_survived_age, bins=bins, label=survived_female_label, edgecolor='black')
plt.xlabel('Female Ages')
plt.ylabel('Number of Female Passengers')
plt.title('Female Passengers and Survived')
plt.legend(loc='upper right')
plt.savefig('Female Passengers and Survived.png')

male_class = [int(c) for c in male_class]
male_survived_class = [male_class[i] for i in range(len(male_class)) if male_survived[i] == '1']

female_class = [int(c) for c in female_class]
female_survived_class = [female_class[i] for i in range(len(female_class)) if female_survived[i] == '1']

bins = 3

plt.figure()
plt.hist(male_class, bins=bins, edgecolor='black')
plt.hist(male_survived_class, bins=bins, edgecolor='black')
plt.xlabel('Class')
plt.ylabel('Number of Male Passengers')
plt.title('Male Cabin Classes and Survived')
plt.xticks([1, 2, 3])
plt.savefig('Male Cabin Classes and Survived.png')

plt.figure()
plt.hist(female_class, bins=bins, edgecolor='black')
plt.hist(female_survived_class, bins=bins, edgecolor='black')
plt.xlabel('Class')
plt.ylabel('Number of Female Passengers')
plt.title('Female Cabin Classes and Survived')
plt.xticks([1, 2, 3])
plt.savefig('Female Cabin Classes and Survived.png')

