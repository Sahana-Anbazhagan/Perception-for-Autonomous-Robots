import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""" To read the input data from the csv file """
data = pd.read_csv('data_2.csv')
x = data.iloc[:,0]
y = data.iloc[:,1]

""" This is a scalar value multiplied to the identity matrix to obtain the curve using Least square with Regularization """
R = ([8, 0, 0], [0, 8, 0], [0, 0, 8])

""" 
Calculating the x matrix and determining the transpose of the same to bring it to determine B,
B = inverse((transpose(X)*X) + R) * (transpose(X)*Y)
where X is the coefficient matrix
      Y is the data obtained from the csv file
      R = aI, a is any scalar value and I is an identity matrix
      B is the final expected matrix that will give us the coefficients of the equation
"""
x_s = np.power(x,2)
x_m = np.transpose([x_s, x, np.ones(np.shape(x))])
#print(x_m)

"""
From B we obtained coefficients a, b, c in the equation y = a * (x**2) + b * x + c
We obtain the coeffients as follows:
a = -2.29745229e-03  b = 1.14536716e+00 c = -4.04561982e+01
"""
B = np.matmul(np.linalg.inv(np.matmul(np.transpose(x_m), x_m) + R),np.matmul(np.transpose(x_m),y))
print(B)

"""
The final equation for the parabolic dataset is obtained as Y = XB, since B is a 1x3 matrix in the previous line we
take the transpose of B to multiply it with the same
"""
y_new = np.matmul(x_m,np.transpose(B))

"""
Plotting and displaying the original dataset and the plotting the best fit line
"""
plt.scatter(x, y, label = 'XY_data')
plt.plot(x, y_new, 'r', label = 'Curve Fitting')
plt.legend()
plt.show()