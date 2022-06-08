from sys import float_info # Threshold smallest positive floating value
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
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


N = 10000

mu = np.array([[-.5, -.5, -.5],
               [1, 1, 1]])  # Gaussian distributions means

Sigma = np.array([[[1, -0.5, 0.3],
                   [-0.5, 1, -0.5],
                   [0.3, -0.5, 1]
                   ],

                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]]
                  ])  # Gaussian distributions covariance matrices
n = mu.shape[1]

# Class priors
priors = np.array([0.65, 0.35])
C = len(priors)

# Caculate threshold rule
Lambda = np.ones((C, C)) - np.identity(C)
gamma = (Lambda[1, 0] - Lambda[0, 0]) / \
    (Lambda[0, 1] - Lambda[1, 1]) * priors[0] / priors[1]
print(f'Threshold value: {gamma}')


u = np.random.rand(N)
#threshold = np.linspace(0,10,100)


# Output samples and labels
X = np.zeros([N, n])
labels = np.zeros(N)  # KEEP TRACK OF THIS

# Plot for original data and their true labels
labels = np.random.rand(N) >= priors[0]
L = np.array(range(C))
Nl = np.array([sum(labels == l) for l in L])
print("Number of samples from Class 1: {:d}, Class 2: {:d}".format(
    Nl[0], Nl[1]))

X = np.zeros((N, n))
X[labels == 0, :] = multivariate_normal.rvs(mu[0], Sigma[0], Nl[0])
X[labels == 1, :] = multivariate_normal.rvs(mu[1], Sigma[1], Nl[1])


# Generate ROC curve samples

def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))

    sorted_score = sorted(discriminant_score)

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] +
            sorted_score +
            [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    ind10 = [np.argwhere((d == 1) & (label == 0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d == 1) & (label == 1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]

    ind01 = [np.argwhere((d == 0) & (label == 1)) for d in decisions]
    p01 = [len(inds)/Nlabels[1] for inds in ind01]

    # To find the best value for gamma from the dataset (not theoretical)
    # Here, we find value with lowest probability or error, and convert
    # taus back from log using exp
    prob_error_erm = np.zeros(len(p01))
    for i in range(len(p10)):
        prob_error_erm[i] = np.array((p10[i], p01[i])).dot(Nlabels.T / N)

    best_gamma = np.exp(taus[np.argmin(prob_error_erm)])

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))
    return roc, taus, best_gamma


def perform_lda(X, mu, Sigma, C=2):
    """  Fisher's Linear Discriminant Analysis (LDA) on data from two classes (C=2).

    In practice the mean and covariance parameters would be estimated from training samples.
    
    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.
        mu: Mean vector [C, n].
        Sigma: Covariance matrices [C, n, n].

    Returns:
        w: Fisher's LDA project vector, shape [n, 1].
        z: Scalar LDA projections of input samples, shape [N, 1].
    """

    mu = np.array([mu[i].reshape(-1, 1) for i in range(C)])
    cov = np.array([Sigma[i].T for i in range(C)])

    # Determine between class and within class scatter matrix
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]

    # Regular eigenvector problem for matrix Sw^-1 Sb
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]

    # Extract corresponding sorted eigenvectors
    U = U[:, idx]

    # First eigenvector is now associated with the maximum eigenvalue, mean it is our LDA solution weight vector
    w = U[:, 0]

    # Scalar LDA projections in matrix form
    z = X.dot(w)

    return w, z


    # Fisher LDA Classifer (using true model parameters)
_, discriminant_score_lda = perform_lda(X, mu, Sigma)

# Estimate the ROC curve for this LDA classifier
roc_lda, tau_lda, BGam = estimate_roc(discriminant_score_lda, labels)

# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR
prob_error_lda = np.array((roc_lda[0, :], 1 - roc_lda[1, :])).T.dot(Nl.T / N)

# Min prob error
min_prob_error_lda = np.min(prob_error_lda)
min_ind = np.argmin(prob_error_lda)

# Display the estimated ROC curve for LDA and indicate the operating points
# with smallest empirical error probability estimates (could be multiple)
fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_lda[0], roc_lda[1], 'b:')
ax_roc.plot(roc_lda[0, min_ind], roc_lda[1, min_ind], 'r.',
            label="Minimum P(Error) LDA", markersize=16)
ax_roc.set_title("ROC Curves for ERM and LDA")
ax_roc.legend()

plt.show()
fig_roc


# Use min-error threshold
decisions_lda = discriminant_score_lda >= tau_lda[min_ind]

# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_lda = np.argwhere((decisions_lda == 0) & (labels == 0))
p_00_lda = len(ind_00_lda) / Nl[0]
# False Positive Probability
ind_10_lda = np.argwhere((decisions_lda == 1) & (labels == 0))
p_10_lda = len(ind_10_lda) / Nl[0]
# False Negative Probability
ind_01_lda = np.argwhere((decisions_lda == 0) & (labels == 1))
p_01_lda = len(ind_01_lda) / Nl[1]
# True Positive Probability
ind_11_lda = np.argwhere((decisions_lda == 1) & (labels == 1))
p_11_lda = len(ind_11_lda) / Nl[1]

# Display LDA decisions
fig = plt.figure(figsize=(10, 10))

# class 0 circle, class 1 +, correct green, incorrect red
plt.plot(X[ind_00_lda, 0], X[ind_00_lda, 1], 'og', label="Correct Class 0")
plt.plot(X[ind_10_lda, 0], X[ind_10_lda, 1], 'or', label="Incorrect Class 0")
plt.plot(X[ind_01_lda, 0], X[ind_01_lda, 1], '+r', label="Incorrect Class 1")
plt.plot(X[ind_11_lda, 0], X[ind_11_lda, 1], '+g', label="Correct Class 1")
plt.show()

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("LDA Decisions (RED incorrect)")
plt.tight_layout()


print("Smallest P(error) for ERM = {}".format(prob_error_erm))
print("Smallest P(error) for LDA = {}".format(min_prob_error_lda))
