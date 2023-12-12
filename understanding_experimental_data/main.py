import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def getData(fileName):
    dataFile = open(fileName, 'r')
    x = []
    y = []
    dataFile.readline() #ignore header
    for line in dataFile:
        d, m = line.split(' ')
        x.append(float(d))
        y.append(float(m))
    dataFile.close()
    return (y, x)


x, y = getData('oddExperiment.txt')

linear_coeffs, linear_residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
linear_fit = np.poly1d(linear_coeffs)
quad_coeffs, quad_residuals, _, _, _ = np.polyfit(x, y, 2, full=True)
quad_fit = np.poly1d(quad_coeffs)

linear_r2 = r2_score(y, linear_fit(x))
quad_r2 = r2_score(y, quad_fit(x))

linear_mse = mean_squared_error(y, linear_fit(x))
quad_mse = mean_squared_error(y, quad_fit(x))

plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, linear_fit(x), 'r-', label=f'Fit of degree 1, R2 = {linear_r2:.5f}, LSE = {linear_mse:.5f}')
plt.plot(x, quad_fit(x), 'g-', label=f'Fit of degree 2, R2 = {quad_r2:.5f}, LSE = {quad_mse:.5f}')
plt.legend()
plt.title('oddExperiment Data')
plt.savefig('p1.png')


degrees = [1, 2, 4, 8, 16]
plt.figure()
plt.scatter(x, y, label='Data')

for degree in degrees:
    coeffs, residuals, _, _, _ = np.polyfit(x, y, degree, full=True)
    fit = np.poly1d(coeffs)
    r2 = r2_score(y, fit(x))
    plt.plot(x, fit(x), label=f'Fit of degree {degree}, R2 = {r2:.5f}')

plt.legend()
plt.title('oddExperiment Data')
plt.savefig('p2.png')


xt, yt = getData('TestDataSet.txt')

coeffs_16, _, _, _, _ = np.polyfit(x, y, 16, full=True)
fit_16 = np.poly1d(coeffs_16)
yt_fit_16 = fit_16(xt)

coeffs_2, _, _, _, _ = np.polyfit(x, y, 2, full=True)
fit_2 = np.poly1d(coeffs_2)
yt_fit_2 = fit_2(xt)

r2_score_16 = r2_score(yt, yt_fit_16)
r2_score_2 = r2_score(yt, yt_fit_2)

plt.figure()
plt.scatter(xt, yt, label='TestDataSet')
plt.plot(xt, yt_fit_16, label=f'Fit of degree 16 of oddExperiment Data, R2 = {r2_score_16:.5f}')
plt.plot(xt, yt_fit_2, label=f'Fit of degree 2 of oddExperiment Data, R2 = {r2_score_2:.5f}')
plt.legend()
plt.title('TestDataSet V.S. oddExperiment Data')
plt.savefig('p3.png')