import numpy as np
import cmath
import math
from numpy import linalg as LA
from numpy.linalg import multi_dot

#Point Correspondences
x1=5
x2=150
x3=150
x4=5
y1=5
y2=5
y3=150
y4=150
xp1=100
xp2=200
xp3=220
xp4=100
yp1=100
yp2=80
yp3=80
yp4=200

#A Matrix
A = ([-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
 [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
 [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
 [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
 [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
 [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
 [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
 [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4])

print("")
print("SVD")
print("")

#U matrix
A_transpose = np.transpose(A)
AAT = np.dot(A,A_transpose)
lamAAT,V_AAT = np.linalg.eig(AAT)
V_AAT = np.around(V_AAT, decimals = 4)
lamAAT1 = [0,0,0,0,0,0,0,0]
for i in range(8):
 lamAAT1[i] = math.sqrt(lamAAT[i])
lamAAT1.sort(reverse = True) #sorting Eigen values in descending order


#V matrix
ATA = np.dot(A_transpose,A)
lamATA, V_ATA = np.linalg.eig(ATA)
V_ATA = np.around(V_ATA, decimals = 4)



#Sigma Matrix
sig = np.diag(lamAAT1)
last_column = np.zeros((8,1))
sig = np.append(sig,last_column,axis=1)
sig = np.around(sig, decimals = 4)


#SVD

U_sig = np.dot(V_AAT,sig)
U_sig = np.around(U_sig, decimals = 1)


VT = np.transpose(V_ATA)
A_res = np.dot(U_sig,VT)


print("")
V_ATA_N = -V_ATA
V_ATA_N[:,4] = -V_ATA_N[:,4]
V_ATA_N[:,6] = -V_ATA_N[:,6]
V_ATA_N[:,7] = -V_ATA_N[:,7]
print("Final V")
print("")
print(V_ATA_N)
print("")

#Homography matrix
H = V_ATA[:,8]


print("Checking for AX = 0")
print("A.X = ")
print(np.dot(A,H))
print("")
H = H.reshape(3,3)
print("")
print("Homography Matrix = ")
print("")
print(H)
print("")

dummy = V_AAT[:,6]
V_AAT[:,2] = -V_AAT[:,2]
V_AAT[:,[6, 7]] = V_AAT[:,[7, 6]]
print("")
print("Final U")
print("")
print(V_AAT)
print("")


print("Final sig")
print("")
print(sig)

temp = np.dot(V_AAT, sig)
temp = np.around(temp, decimals = 1)
RES = np.dot(temp, np.transpose(V_ATA_N))
print("")
print("")
print("Result of SVD")
print("")
print("Multiplying U,Sigma and V transpose to generate the original A Matrix")
RES = np.around(RES, decimals = 1)
print("")
print(RES)





