#Cuvre Fiting using Total Least Square for Dataset1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" To read and plot the input data from the csv file """
data = pd.read_csv('data_1.csv')
x_axis = data.iloc[:,0]
y_axis = data.iloc[:,1]

x_bar=0
x_sq_bar=0
y_bar=0


for j in range(250):
    x_bar+=x_axis[j]
    y_bar+=y_axis[j]
    x_sq_bar+=np.power(x_axis[j],2)
U = np.zeros((250,3))
for i in range(250):
    U[i][0]=np.power(x_axis[i],2)-(x_sq_bar/250)
    U[i][1]=np.power(x_axis[i],1)-(x_bar/250)
    U[i][2]=np.power(y_axis[i],1)-(y_bar/250)

capU = np.dot(np.transpose(U),U)
g =[0.0015, -0.7360,  0.6770]

V = np.dot(np.transpose(capU),capU)


lamATA, V_ATA = np.linalg.eig(V)
V_ATA = np.around(V_ATA, decimals = 9)

G=V_ATA[:,2]


a = g[0]
b = g[1]
c = g[2]

d = (a*x_sq_bar/250)+(b*x_bar/250)+(c*y_bar/250)
new_y=(d/c)-(a*np.power(x_axis,2)/c) - (b*x_axis/c)


plt.scatter(x_axis,y_axis,label = 'Given_data')
plt.plot(x_axis,new_y,"r",label = 'TLS_Fitting')
plt.legend()
plt.show()