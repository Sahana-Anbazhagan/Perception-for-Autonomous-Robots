import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" To read and plot the input data from the csv file """
data = pd.read_csv('data_1.csv')
x_axis = data.iloc[:,0]
y_axis = data.iloc[:,1]

"""
Building the model for this dataset
Calculating the A and B matrices to determine X,
B = inverse((transpose(X)*X)) * (transpose(X)*Y)
For a simple square matrix we can calculate the same as,
X = inverse(A)*B
where A is the coefficient matrix (consists of the x^n terms obtained from the csv file)
      Y is the data obtained from the csv file
      X is the final expected matrix that will give us the coefficients of the equation (unknowns)
"""
x1 = sum(x_axis)
x2 = sum(np.power(x_axis, 2))
x3 = sum(np.power(x_axis, 3))
x4 = sum(np.power(x_axis, 4))
xy = sum(x_axis * y_axis)
x2y = sum(np.power(x_axis, 2) * y_axis)
y = sum(y_axis)
A = ([x4,x3,x2], [x3, x2, x1], [x2, x1, 250])
B = ([x2y, xy, y])
A_inv = np.linalg.inv(A)
X = np.matmul(A_inv, B)

""" Printing the obtained values of the coefficients of the equation a, b, c """
print("The value of a is", X[0])
print("The value of b is", X[1])
print("The value of c is", X[2])


""" Plotting the original graph along with the curve fitted graph """
Y_pred = X[0] * (np.power(x_axis, 2)) + X[1] * x_axis + X[2] # The final equation of a parabola
#print(Y_pred) # To print the values of the above equation
plt.scatter(x_axis,y_axis, label = 'Initial dataset') # Graph with only the data points
plt.plot(x_axis, Y_pred, 'red', label = 'Curve Fitting') # Graph with the curve fitted to the data
plt.legend()
plt.show()
