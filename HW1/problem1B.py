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

mu = np.array([[-.5, -.5,-.5],
                [1, 1, 1]])  # Gaussian distributions means

Sigma = np.array([[[1, -0.5, 0.3],
                   [-0.5, 1, -0.5],
                   [0.3, -0.5, 1]
                  ],
                   
                  [[1, 0.3, -0.2],
                   [0.3, 1, 0.3],
                   [-0.2, 0.3, 1]]
                  ])  # Gaussian distributions covariance matrices


Sigma = np.array([[[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]
                  ],

                  [[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]
                  ]]
                  )
n = mu.shape[1]

# Class priors
priors = np.array([0.65, 0.35])
C = len(priors)

# Caculate threshold rule
Lambda = np.ones((C, C)) - np.identity(C)
gamma = (Lambda[1,0] - Lambda[0,0])/(Lambda[0,1] - Lambda[1,1]) * priors[0] / priors[1]
print(f'Threshold value: {gamma}')


u = np.random.rand(N)
#threshold = np.linspace(0,10,100)


# Output samples and labels
X = np.zeros([N, n])
labels = np.zeros(N) # KEEP TRACK OF THIS

# Plot for original data and their true labels
labels = np.random.rand(N) >= priors[0]
L = np.array(range(C))
Nl = np.array([sum(labels == l) for l in L])
print("Number of samples from Class 1: {:d}, Class 2: {:d}".format(Nl[0], Nl[1]))

X = np.zeros((N, n))
X[labels == 0, :] =  multivariate_normal.rvs(mu[0], Sigma[0], Nl[0])
X[labels == 1, :] =  multivariate_normal.rvs(mu[1], Sigma[1], Nl[1])

# # Plot the original data and their true labels
# fig = plt.figure(figsize=(10, 10))
# plt.plot(X[labels==0, 0], X[labels==0, 1], 'bo', label="Class 0")
# plt.plot(X[labels==1, 0], X[labels==1, 1], 'k+', label="Class 1")

# plt.legend()
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")
# plt.title("Data and True Class Labels")
# plt.tight_layout()
# plt.show()



# Expected Risk Minimization Classifier (using true model parameters)
# In practice the parameters would be estimated from training samples
# Using log-likelihood-ratio as the discriminant score for ERM
class_conditional_likelihoods = np.array([multivariate_normal.pdf(X, mu[l], Sigma[l]) for l in L])
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])
print("class conditional: ",discriminant_score_erm)


#class_priors = np.diag(priors)
#class_posteriors = class_priors.dot(class_conditional_likelihoods)
#cond_risk = Lambda.dot(class_posteriors)
#decisions1 = np.argmin(cond_risk, axis=0)



# Gamma threshold for MAP decision rule (remove Lambdas and you obtain same gamma on priors only; 0-1 loss simplification)
gamma_map = (Lambda[1,0] - Lambda[0,0]) / (Lambda[0,1] - Lambda[1,1]) * priors[0]/priors[1]
# Same as:
# gamma_map = priors[0]/priors[1]
# print(f'Gamma threshold (Decision Rule): {gamma_map}')



from sys import float_info # Threshold smallest positive floating value

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

    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]

    ind01 = [np.argwhere((d==0) & (label==1)) for d in decisions]
    p01 = [len(inds)/Nlabels[1] for inds in ind01]

    # To find the best value for gamma from the dataset (not theoretical)
    # Here, we find value with lowest probability or error, and convert 
    # taus back from log using exp
    prob_error_erm = np.zeros(len(p01))
    for i in range(len(p10)):
        prob_error_erm[i] = np.array((p10[i], p01[i])).dot(Nlabels.T / N)


    best_gamma = np.exp(taus[np.argmin(prob_error_erm)])
    p_error_erm = min(prob_error_erm)

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11))
    return roc, taus, best_gamma, p_error_erm


# # Gamma threshold for MAP decision rule (remove Lambdas and you obtain same gamma on priors only; 0-1 loss simplification)
# trials = 101
# threshold = np.linspace(0,10,trials)
gamma_map = priors[0]/priors[1]

decisions_map = discriminant_score_erm >= np.log(gamma_map)
# Get indices and probability estimates of the four decision scenarios:
# (true negative, false positive, false negative, true positive)

# True Negative Probability
ind_00_map = np.argwhere((decisions_map==0) & (labels==0))
p_00_map = len(ind_00_map) / Nl[0]
# False Positive Probability
ind_10_map = np.argwhere((decisions_map==1) & (labels==0))
p_10_map = len(ind_10_map) / Nl[0]
# False Negative Probability
ind_01_map = np.argwhere((decisions_map==0) & (labels==1))
p_01_map = len(ind_01_map) / Nl[1]
# True Positive Probability
ind_11_map = np.argwhere((decisions_map==1) & (labels==1))
p_11_map = len(ind_11_map) / Nl[1]
# Probability of error for MAP classifier, empirically estimated
prob_error_erm = np.array((p_10_map, p_01_map)).dot(Nl.T / N)



# Construct the ROC for ERM by changing log(gamma)
roc_erm, _, bestGamma, p_error_erm = estimate_roc(discriminant_score_erm, labels)
roc_map = np.array((p_10_map, p_11_map))

#roc_erm, taus = ROC_Curve(discriminant_score_erm, labels)

fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
ax_roc.plot(roc_erm[0], roc_erm[1])
ax_roc.plot(roc_map[0], roc_map[1], 'rx', label="Minimum P(Error) MAP", markersize=16)
ax_roc.legend()
ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
plt.title('ROC Curve for Naive Bayes')
plt.grid(True)
plt.show()

print('Gamma MAP (Theoretical): ', gamma_map)
print('Probability of Error: ', prob_error_erm)
print('Best Gamma (Based on Data): ', bestGamma)
print('Probability of Error(Empirical): ', p_error_erm)





