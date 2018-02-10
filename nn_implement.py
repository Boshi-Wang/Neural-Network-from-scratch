import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import sklearn
import matplotlib
import time

def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

def reLU(x):
    return np.maximum(0,x)
def reLU_derivative(x):
    return np.maximum(0,x)/x

# change default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

X_, y = sklearn.datasets.make_moons(200, noise=0.20)

# normalize data to mean 0 and std 1
mean = np.mean(X_,axis=0)
std = np.std(X_,axis=0)
X = (X_-mean)/std

plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

class neural_network:
    def __init__(self, layers=[2,5,5,1], activation='tanh'):
        self.layers = layers
        if activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative
        if activation == 'reLU':
            self.activation = reLU
            self.activation_deriv = reLU_derivative

        # initialize weights and biases
        self.W = [None]
        self.b = [None]
        for i in range(len(self.layers)-1):
            m = self.layers[i]
            n = self.layers[i+1]
            weight = 0.25*(np.random.random((n, m)) * 2 - 1)
            bias = 0.05*(np.random.random((n, 1)) * 2 - 1)
            self.W.append(weight)
            self.b.append(bias)

    def fit(self, X_, y_, learning_rate=0.2, batch_size=10):
        num_examples = X_.shape[0]

        # find a batch
        indices = np.random.randint(num_examples, size=batch_size)
        X = X_[indices]
        y = y_[indices].reshape(1,batch_size)

        # forward propagation
        a=[X.T]
        z=[None]
        for i in range(1,len(self.W)-1):
            z.append(np.dot(self.W[i],a[i-1])+self.b[i])
            a.append(self.activation(z[-1]))
        # output layer, using logistic function
        i += 1
        z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
        a.append(logistic(z[-1]))
        assert(a[-1].shape == (self.layers[-1],batch_size))

        # back prop
        da,dz,dW,db=[],[],[],[]
        da.append(-y/a[-1]+(1-y)/(1-a[-1]))
        for i in range(len(a)-1):
            if i == 0:
                f = logistic_derivative
            else:
                f = self.activation_deriv

            dz.append(da[-1]*f(z[-(i+1)]))
            dW.append(1/batch_size * np.dot(dz[-1],a[-(i+2)].T))
            db.append(1/batch_size * np.sum(dz[-1],axis=1,keepdims=True))
            da.append(np.dot(self.W[-(i+1)].T,dz[-1]))

        da.reverse()
        dz.reverse()
        dW.reverse()
        dW.insert(0,None)
        db.reverse()
        db.insert(0,None)
        assert(len(dW)==len(self.W)==len(db)==len(self.b))

        # update
        for i in range(1,len(dW)):
            self.W[i] -= learning_rate*dW[i]
            self.b[i] -= learning_rate*db[i]

    def compute_cost(self,X,y):
        # forward propagation
        a = [X.T]
        z = [None]
        for i in range(1, len(self.W) - 1):
            z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
            a.append(self.activation(z[-1]))

        # output layer, using logistic function
        i += 1
        z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
        a.append(logistic(z[-1]))
        cost = (- 1 / X.shape[0]) * np.sum(y * np.log(a[-1]) + (1 - y) * (np.log(1 - a[-1])))  # compute cost

        return cost
    def predict(self,X):
        # forward propagation
        a = [X.T]
        z = [None]
        for i in range(1, len(self.W) - 1):
            z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
            a.append(self.activation(z[-1]))

        # output layer, using logistic function
        i += 1
        z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
        a.append(logistic(z[-1]))
        b=[]
        for i in range(len(a[-1][0])):
            if a[-1][0][i] >= 0.5:
                b.append(1)
            else:
                b.append(0)
        b=np.array(b)
        return b

    def compute_accuracy(self,X,y):
        # forward propagation
        a = [X.T]
        z = [None]
        for i in range(1, len(self.W) - 1):
            z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
            a.append(self.activation(z[-1]))

        # output layer, using logistic function
        i += 1
        z.append(np.dot(self.W[i], a[i - 1]) + self.b[i])
        a.append(logistic(z[-1]))
        b = []
        for i in range(len(a[-1][0])):
            if a[-1][0][i] >= 0.5:
                b.append(1)
            else:
                b.append(0)
        b = np.array(b)
        return np.sum(b == y) / y.shape[0]

# main
nn = neural_network(activation='tanh')
print("initial cost: "+str(nn.compute_cost(X,y)))
tic = time.time()
for i in range(100000):
    if i % 1000 == 0:
        print("cost after "+str(i)+" iterations: "+str(nn.compute_cost(X,y)))
    nn.fit(X, y)
toc = time.time()
print("training time："+str((toc-tic))+" s")
print("Optimized cost: "+str(nn.compute_cost(X,y)))
print("accuracy on training set: "+str(nn.compute_accuracy(X,y)))

plot_decision_boundary(lambda x: nn.predict(x))
plt.show()