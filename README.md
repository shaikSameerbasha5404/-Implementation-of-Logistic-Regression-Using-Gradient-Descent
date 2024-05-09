# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Obtain the graph.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Shaik Sameer Basha
RegisterNumber:  212222240093
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
*/

*/
```

## Output:
Array Value of x:

![5 1](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/ad1c5212-e920-4580-a2a4-3463778e0d44)



Array Value of y:


![5 2](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/37fa891e-a516-449e-9aa8-a07ff50d2837)


Exam 1 - score graph:


![5 3](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/48c5d2cb-289e-4445-8532-a7127536513f)


Sigmoid function graph:


![5 4](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/8cad721c-30e5-4ad0-8aca-ca49a3c461ed)


X_train_grad value:


![5 5](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/251a9793-3705-4d42-9a03-9deb5e39d588)


Y_train_grad value:

![5 6](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/fb575892-11d3-4f07-bd12-327150853400)



Print res.x:


![5 7](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/2bb7f3a5-963c-453b-96f9-f4aaac1619e0)


Decision boundary - graph for exam score:


![5 8](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/bf053865-c012-4a79-a127-123242afc133)


Proability value:

![5 9](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/673df98b-5c77-4812-b26b-97f90bb7d054)



Prediction value of mean:


![5 10](https://github.com/shaikSameerbasha5404/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118707756/5d5eb1b6-714c-4e0f-a54d-dbac325e9682)


























## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

