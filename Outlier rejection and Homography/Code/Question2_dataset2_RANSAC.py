#Implementing RANSAC on dataset2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_2.csv')
x_axis = data.iloc[:,0]
y_axis = data.iloc[:,1]
thr=np.zeros((250,1))
n_inliers=np.zeros((250,1))
best_inlier_count=0
for i in range(80):
    no_of_inliers=0
    x=np.random.choice(x_axis,3)
    y=[y_axis[x[0]/2],y_axis[x[1]/2],y_axis[x[2]/2]]
    A = np.zeros((3,3))
    B = np.zeros((3,1))
    for j in range(3):
        A[0][0]+=np.power(x[j],4)
        A[0][1]+=np.power(x[j],3)
        A[0][2]+=np.power(x[j],2)
        A[1][2]+=x[j]
        B[0][0]+=np.power(x[j],2)*y[j]
        B[1][0]+=x[j]*y[j]
        B[2][0]+=y[j]
    A[1][0]=A[0][1]
    A[1][1]=A[0][2]
    A[2][0]=A[1][1]
    A[2][1]=A[1][2]
    A[2][2]=3
    x_i = np.dot(np.linalg.inv(A),B)
    threshold=0
    y_est=np.zeros((250,1))
    for k in range(250):
        y_est[k] = x_i[0]*np.power(x_axis[k],2)+x_i[1]*x_axis[k]+x_i[2]
        threshold+=abs(y_axis[k]-y_est[k])
    thr[i]=threshold/250
    for l in range(250):
        if (abs(y_axis[l]-y_est[l])+5 > thr[i] and abs(y_axis[l]-y_est[l])-10 < thr[i]):
            no_of_inliers+=1
    n_inliers[i]=no_of_inliers;
    print("No. of inliers for this case" + str(n_inliers[i]))
    if (best_inlier_count<n_inliers[i]):
        best_inlier_count=n_inliers[i]
        best_x_i=x_i     
        
#The next four lines can be easily hidden to make it look less cluttered
#but for our understanding, we have printed plots for each of the iterations.
    y = x_i[0]*np.power(x_axis,2)+x_i[1]*np.power(x_axis,1)+x_i[2]
    plt.scatter(x_axis,y_axis)
    plt.plot(x_axis,y,'r')
    plt.show()
    
print("The best fit that we got till now:")
print("Maximum number of inliers: " + str(best_inlier_count))
best_y = best_x_i[0]*np.power(x_axis,2)+best_x_i[1]*np.power(x_axis,1)+best_x_i[2]
plt.scatter(x_axis,y_axis)
plt.plot(x_axis,best_y,'r')
plt.show()
