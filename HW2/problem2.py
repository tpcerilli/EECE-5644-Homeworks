import matplotlib.pyplot as plt # For general plotting

import numpy as np

from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix

from modules import prob_utils
from homework2 import hw2q2
from math import ceil, floor 
from sklearn.preprocessing import PolynomialFeatures

np.set_printoptions(suppress=True)

np.random.seed(7)      # seed 7 is really bad for quadratic

plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=18)   # fontsize of the figure title



# Breaks the matrix X and vector y into batches
def batchify(X, y, batch_size, N):
    X_batch = []
    y_batch = []

    # Iterate over N in batch_size steps, last batch may be < batch_size
    for i in range(0, N, batch_size):
        nxt = min(i + batch_size, N + 1)
        X_batch.append(X[i:nxt, :])
        y_batch.append(y[i:nxt])

    return X_batch, y_batch


def gradient_descent(loss_func, theta0, X, y, N, *args, **kwargs):
    # Mini-batch GD. Stochastic GD if batch_size=1.

    # Break up data into batches and work out gradient for each batch
    # Move parameters theta in that direction, scaled by the step size.

    # Options for total sweeps over data (max_epochs),
    # and parameters, like learning rate and threshold.

    # Default options
    max_epoch = kwargs['max_epoch'] if 'max_epoch' in kwargs else 200
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    epsilon = kwargs['tolerance'] if 'tolerance' in kwargs else 1e-6

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10

    # Turn the data into batches
    X_batch, y_batch = batchify(X, y, batch_size, N)
    num_batches = len(y_batch)
    print("%d batches of size %d:" % (num_batches, batch_size))

    theta = theta0
    m_t = np.zeros(theta.shape)

    trace = {}
    trace['loss'] = []
    trace['theta'] = []

    # Main loop:
    for epoch in range(1, max_epoch + 1):
        # print("epoch %d\n" % epoch)
        
        loss_epoch = 0
        for b in range(num_batches):
            X_b = X_batch[b]
            y_b = y_batch[b]
            # print("epoch %d batch %d\n" % (epoch, b))

            # Compute NLL loss and gradient of NLL function
            loss, gradient = loss_func(theta, X_b, y_b, *args)
            loss_epoch += loss
            
            # Steepest descent update
            theta = theta - alpha * gradient
            
            # Terminating Condition is based on how close we are to minimum (gradient = 0)
            if np.linalg.norm(gradient) < epsilon:
                print("Gradient Descent has converged after {} epochs".format(epoch))
                break
        # Storing the history of the parameters and loss values per epoch
        trace['loss'].append(np.mean(loss_epoch))
        trace['theta'].append(theta)
        # print(trace['loss'])
        # Also break epochs loop
        if np.linalg.norm(gradient) < epsilon:
            break

    return theta, trace


def cubic_transformation(X):
    n = X.shape[1]
    phi_X = X
    
    # Take all monic polynomials for a quadratic
    phi_X = np.column_stack((phi_X, X[:, 1] * X[:, 1],              X[:, 1] * X[:, 2],              X[:, 2] * X[:, 2],              \
                                    X[:, 1] * X[:, 1] * X[:, 1],    X[:, 1] * X[:, 1] * X[:, 2],    X[:, 1] * X[:, 2] * X[:, 2],    \
                                    X[:, 2] * X[:, 2] * X[:, 2]
                                    ))

    return phi_X


def plot3(a, b, c, name="Training", mark="o", col="y"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    # ax.set_prop_cycle(color=['red', 'green', 'blue'])
    # ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    # plt.show()

def lin_reg_loss(theta, X, y):
    # Size of batch
    B = X.shape[0]
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    # print(error)
    # loss_f = np.mean(error ** 2)

    # loss_f = 0.5*error.T.dot(error)
    loss_f = (X.dot(theta)-y).T.dot(X.dot(theta)-y)
    # print(loss_f)
    # Partial derivative for GD, X^T * ((X * theta) - y)
    g = (1 / B) * X.T.dot(error)
    # g = (X.T.dot(error) - X.T.dot(y))

    return loss_f, g


def MAP_gamma(X,y,gamma):
    theta = np.linalg.inv(X.T.dot(X)+gamma*np.identity(X.shape[1])).dot(X.T.dot(y))
    return theta

def mean_square_err(X,y,theta):
    y_predict = X.dot(theta) #+ noiseV
    ### MSE
    mse = np.mean((y - y_predict)**2)
    return mse

"""
cubic polynomial y = c(x,theta) + v
           where v = Gauss(0,sigma**2)
x = [1,x1,x2,x1x1,x1x2,x2x2,x1x1x1,x1x1x2,x1x2x2,x2x2x2]
10 terms (including bias)
"""

# Options for mini-batch gradient descent
opts = {}
opts['max_epoch'] = 100
opts['alpha'] = 1e-6
opts['tolerance'] = 1e-3

opts['batch_size'] = 10



def main():
    # mu = np.array([[0,0,0,0,0,0,0,0,0,0]])
    mu = np.zeros(10)
    sigma2 = 1
    sigma = np.identity(10)*sigma2
    mu = 0
    sigma = 1

  
    Ntrain = 100
    Nvalidate = 1000
    # xTrain= hw2q2.generateData(Ntrain)
    # xVal = hw2q2.generateData(Nval)
    xTrain, yTrain, xValidate, yValidate = hw2q2.hw2q2()
    noiseT = multivariate_normal.rvs(mu,sigma,Ntrain)
    noiseV = multivariate_normal.rvs(mu,sigma,Nvalidate)
    # shuffled_indices = np.random.permutation(N[i]) 
    # # Shuffle row-wise X (i.e. across training examples) and labels using same permuted order
    # Xshuf = data[i][shuffled_indices]
    # yshuf = labels[i][shuffled_indices]

    xAugT = np.column_stack((np.ones(Ntrain), xTrain)) 
    yAug = np.column_stack((np.ones(Ntrain), yTrain)) 
    X3train = cubic_transformation(xAugT) #+ noiseT

    xAugV = np.column_stack((np.ones(Nvalidate), xValidate)) 
    X3validate = cubic_transformation(xAugV) #+ noiseT

    # poly = PolynomialFeatures(3)
    # X3 = poly.fit_transform(xTrain)
    # # print(xTrain[0])
    # print(X3[0])

    nCubic = X3train.shape[1]
    theta0 = np.random.randn(nCubic)
    # theta0 = np.zeros(nCubic)

    theta_gd, trace = gradient_descent(lin_reg_loss, theta0, X3train, yTrain, Ntrain, **opts)
    theta_MAP = MAP_gamma(X3train,yTrain,0)

    #Results
    print('theta start:')
    print(theta0)
    print('theta MLE:')
    print(theta_gd)
    print('theta MAP:')
    print(theta_MAP)
    print()
    # print("Mini-batch GD Theta: ", theta_gd)
    # print("MSE: ", trace['loss'][-1])

    ### Now compare to test data and calc accuracy using Mean Square Error ###
    
    mse_gd = mean_square_err(X3validate,yValidate,theta_gd)
    mse_MAP = mean_square_err(X3validate,yValidate,theta_MAP)
    print('MSE GD:', mse_gd)
    print('MSE MAP:', mse_MAP)

    # y = c + noise

    # X_gd = multivariate_normal.rvs(theta_gd,sigma,Nvalidate) + noiseV
    # X_MAP = multivariate_normal.rvs(theta_MAP,sigma,Nvalidate) + noiseV
    # print()
    y_MAP = X3validate.dot(theta_MAP) + noiseV
    y_gd = X3validate.dot(theta_gd) + noiseV





    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xValidate[:, 0], xValidate[:, 1], yValidate, marker='o', color='b', label='True Data')
    ax.scatter(X3validate[:, 1], X3validate[:, 2], y_gd, marker='o', color='y', label='ML Estimate')
    ax.scatter(X3validate[:, 1], X3validate[:, 2], y_MAP, marker='o', color='r', label='MAP Estimate')
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    ax.legend()
    plt.show()

    #####################################
    ###         Varying gamma         ###               
    #####################################

    trials = 10001
    gamma = np.linspace(0.0001,1000,trials)
    # print(gamma)
    mse_range = []
    for i in range(trials):
        theta_temp = MAP_gamma(X3train,yTrain,gamma[i])
        mse_range.append(mean_square_err(X3validate,yValidate,theta_temp))
    plt.plot(gamma,mse_range)
    plt.title("MSE vs gamma")
    plt.xlabel('gamma')
    plt.ylabel('MSE')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()

    return

if __name__ == '__main__':
    main()