import matplotlib.pyplot as plt
import numpy as np

N = 50
sigma = 1
t_beta0 = 15
t_beta1 = -5

np.random.seed(0)

x = np.linspace(1,10,50)
y = x*t_beta1+t_beta0+np.random.randn(N)*sigma

fig1 = plt.figure("Figure 1")
plt.scatter(x,y,c='red',s=10) #visualize the synthetic data set

X = np.c_[np.ones([x.shape[0],1]),x] #create matrix X with first column all ones and second column as arrary x
y_c = y[:,np.newaxis] #turn row vector y into colum vector y_c

XTX = np.dot(X.T,X)
XTX_inv = np.linalg.inv(XTX)
W = np.linalg.multi_dot([XTX_inv,X.T,y_c]) #calculate the weight vector using the formula (X^TX)^(-1)X^Ty

beta0 = W[0]
beta1 = W[1]

y_hat = beta1*x+beta0

residual = y_hat-y

SSE = np.sum(residual**2)

print("The SSE error of the fitted line is: %f"%SSE)
print("\nbeta0 is ",beta0)
print("\nbeta1 is ",beta1)

plt.plot(x,y_hat)
plt.title("linear least square fitting")
plt.xlabel("x")
plt.ylabel("y")


#now let's use the linear regression model from sklearn package to fit the given data set
from sklearn.linear_model import LinearRegression
lrg = LinearRegression()
lrg.fit(X,y_c)

#print(lrg.coef_.shape)
beta1_lrg = lrg.coef_[0,1]
beta0_lrg = lrg.intercept_

y_pred = lrg.predict(X)
residual_lrg = y_pred-y_c
SSE_lrg = np.sum(residual_lrg**2)

fig2 = plt.figure("Figure 2")
plt.scatter(x,y,c="red",s=10)
plt.plot(x,y_pred,c="green")
plt.title("liear square fit using sklearn")
plt.xlabel("x")
plt.ylabel("y")

print("\nThe SSE error using the LinearRegression model is: %f"%SSE_lrg)
print("\nbeta0 estimated using sklearn is ",beta0_lrg)
print("\nbeta1 estimated using sklearn is ",beta1_lrg)