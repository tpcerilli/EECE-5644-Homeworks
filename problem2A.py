import matplotlib.pyplot as plt # For general plotting

import numpy as np

from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title

N = 10000
xm = 5
ym = 1
mu = np.array([ [xm*0, ym*0],
                [xm*1, ym*1],
                [xm*2, ym*2],
                [xm*3, ym*3]])  # Gaussian distributions means

Sigma = np.array([[[7, 0],
                   [0, 7]],
                   
                  [[2, 0],
                   [0, 2]],

                  [[5, 0],
                   [0, 5]],

                  [[3, 0],
                   [0, 3]]
                ])  # Gaussian distributions covariance matrices
n = mu.shape[1]

# Class priors
priors = np.array([0.2, 0.25, 0.35, 0.2])
C = len(priors)

# # Caculate threshold rule
# Lambda = np.ones((C, C)) - np.identity(C)
# gamma = (Lambda[1,0] - Lambda[0,0])/(Lambda[0,1] - Lambda[1,1]) * priors[0] / priors[1]
# print(f'Threshold value: {gamma}')

u = np.random.rand(N)
thresholds = np.cumsum(priors)
thresholds = np.insert(thresholds, 0, 0) # For intervals of classes
print(f'Priors: {thresholds}')

# Output samples and labels
X = np.zeros([N, n])
labels = np.zeros(N) # KEEP TRACK OF THIS


# Plot for original data and their true labels

fig = plt.figure(figsize=(8, 8))
marker_shapes = 'd+.x'
marker_colors = 'rbgy' 

L = np.array(range(1,C+1))

# Create data and plot it
for l in L:
    # Get randomly sampled indices for this component
    indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
    # No. of samples in this component
    Nl = len(indices)  
    labels[indices] = l * np.ones(Nl)
    X[indices, :] =  multivariate_normal.rvs(mu[l-1], Sigma[l-1], Nl)
    plt.plot(X[labels==l, 0], X[labels==l, 1], marker_shapes[l-1] + marker_colors[l-1], label="True Class {}".format(l))

Nl = np.array([sum(labels == l) for l in L])
print("Number of samples from Class 1: {:d}".format(Nl[0]))
print("                       Class 2: {:d}".format(Nl[1]))
print("                       Class 3: {:d}".format(Nl[2]))
print("                       Class 4: {:d}".format(Nl[3]))


plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Class Labels")
plt.tight_layout()
plt.show()




# Min prob. of error classifier
# Conditional likelihoods of each class given x, shape (C, N)
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[c], Sigma[c]) for c in range(C)])
# Take diag so we have (C, C) shape of priors with prior prob along diagonal
class_priors = np.diag(priors)
# class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
# with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
class_posteriors = class_priors.dot(class_cond_likelihoods)

# MAP rule, take largest class posterior per example as your decisions matrix (N, 1)
# Careful of indexing! Added np.ones(N) just for difference in starting from 0 in Python and labels={1,2,3}
decisions = np.argmax(class_posteriors, axis=0) + np.ones(N) 

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Number of Misclassified Samples: {:d}".format(N - correct_class_samples))

# Alternatively work out probability error based on incorrect decisions per class
# perror_per_class = np.array(((conf_mat[1,0]+conf_mat[2,0])/Nl[0], (conf_mat[0,1]+conf_mat[2,1])/Nl[1], (conf_mat[0,2]+conf_mat[1,2])/Nl[2]))
# prob_error = perror_per_class.dot(Nl.T / N)

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))


fig = plt.figure(figsize=(12, 9))

marker_colors = 'rgrg'
marker_shapes = 's.^o'
for r in L: # Each decision option
    for c in L: # Each class label
        ind_rc = np.argwhere((decisions==r) & (labels==c))

        # Decision = Marker Shape; True Labels = Marker Color
        #marker = marker_shapes[r-1] + marker_colors[c-1]
        if r == c:
            plt.plot(X[ind_rc, 0], X[ind_rc, 1], marker_shapes[r-1] + 'g', label="True Class {}".format(c))
        else:
            plt.plot(X[ind_rc, 0], X[ind_rc, 1], marker_shapes[r-1] + 'r', label="Predicted as Class {}".format(r))
#print(decisions)

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Classification Decisions: Marker Shape/Predictions, Color/True Labels")
plt.tight_layout()
plt.show()
#print(len(X))
#print(len(X[0]))