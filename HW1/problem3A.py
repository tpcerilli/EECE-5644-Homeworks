from matplotlib.axis import XAxis
import matplotlib.pyplot as plt # For general plotting

import numpy as np

from scipy.stats import multivariate_normal # MVN not univariate
from sklearn.metrics import confusion_matrix
import csv

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


def main():
    # Read in data

    reader = csv.reader(open("./winequality-white.csv"),delimiter=';')
    data_white = [] #np.array([[]])
    count = 0
    for row in reader:
        #Skip headers
        count += 1
        if (count <= 1):
            header = row
            continue

        data_white.append(list(np.float_(row)))

    #print(header)
    # print(data_white)
    data_white = np.array(data_white)

    def find_mean_std(data):
            
        means = np.zeros(len(header))
        diff = np.zeros(len(header))
        summation = np.zeros(len(header))
        #stdev_red = np.zeros(len(header))
        for i in range(1, len(data)+1):
            for j in range(len(header)):
                diff[j]  = data[i-1][j] - means[j]
                summation[j]  += diff[j] * diff[j] * (i - 1.0) / i
                means[j] += diff[j] / i

        std = np.sqrt(summation / len(data))
        return means, std

    #mean_w,std_w = find_mean_std(data_white)
    #mu,Sigma = find_mean_std(data_white)

    # mu = np.array([mean_r,mean_w])
    # Sigma = np.array([std_r,std_w])

    #n = mu.shape[0]
    N = len(data_white)

    Nquality = np.zeros(11)
    for i in range(N):
        qual = int(data_white[i][11])
        Nquality[qual] += 1
        


    #Labels = ['red','white']
    #print(Nquality)
    priors = Nquality/N
    C = len(priors)
    #print(priors)



    # Lambda = np.ones((C, C)) - np.identity(C)
    # #gamma = (Lambda[1,0] - Lambda[0,0])/(Lambda[0,1] - Lambda[1,1]) * priors[0] / priors[1]
    # gamma = priors[0] / priors[1]
    # print(f'Threshold value: {gamma}')

    u = np.random.rand(N)
    thresholds = np.cumsum(priors)
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes
    #print(f'Priors: {thresholds}')



    # Find mu and Sigma for each class/quality rating
    mu = np.zeros((C,C))
    Sigma = np.zeros((C,C))

    L = np.array(range(C))
    labels = np.zeros(N) # KEEP TRACK OF THIS

    for l in L:
        index = np.argwhere((data_white[:,11] == l))[:,0]
        count = 0
        Nl = len(index)
        labels[index] = l * np.ones(Nl)

        #std = np.zeros(C)
        mean = np.zeros(C)
        diff = np.zeros(C)
        summation = np.zeros(C)

        for ind in index:
            for i in range(C):
                count += 1
                summation[i] += data_white[ind][i]
                diff[i]  = data_white[ind][i] - mean[i]
                summation[i]  += diff[i] * diff[i] * (count - 1.0) / count
                mean[i] += diff[i] / count
                if(summation[i]<0):
                    print('Error: less than zero')
        if (len(index) != 0):
            std = np.sqrt(summation / len(index))
        else:
            std = np.zeros(C)
        mu[l] = mean
        Sigma[l] = np.array([std])
    Sigma_diag = []
    #print("sigma     ",Sigma[5])
    for i in range(C):
        Sigma_diag.append(np.diag(Sigma[i]))
        # Sigma_diag[i] += 0.001*np.identity(C)
    Sigma_np = np.cov(data_white[:,:11],rowvar=False)
    #print(len(Sigma_np))
    
    #print("sigma diag",Sigma_diag[0])
    #     Sigma[l][0] = np.array([std])
    # for i in range(C):
    #     for j in range(1,C):
    #         for k in range(C):
    #             Sigma[i][j][k] = Sigma[i][j-1][k-1]
    n = mu.shape[0]
    Sigma_diag += 0.00001*np.identity(C)

    # Output samples and labels
    X = np.zeros([N, n])

    fig = plt.figure(figsize=(8, 8))
    marker_shapes = 'd+.xd+.xd+.'
    marker_shapes = 'oooooo+++++'
    marker_colors = 'rbgcmykrbgc' 

    # Create data and plot it
    xax = 1
    yax = 9
    # print(L)
    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[l] <= u) & (u <= thresholds[l+1]))[:, 0]
        # No. of samples in this component
        Nl = len(indices)  
        labels[indices] = l * np.ones(Nl)
        X[indices, :] =  multivariate_normal.rvs(mu[l], Sigma_diag[l], Nl)

        plt.plot(data_white[labels==l, xax], data_white[labels==l, yax], marker_shapes[l] + marker_colors[l], label="Quality of {}".format(l))

    Nl = np.array([sum(labels == l) for l in L])
    print(f"Number of samples from Quality Rating 0: {Nl[0]}".format(Nl[0]))
    for l in L:
        if l == 0:
            continue
        print(f"...................... Quality Rating {l}: {Nl[l]}")


    plt.legend()
    #plt.xlim(0,20)
    #plt.ylim(0,300)
    plt.xlabel(header[xax])
    plt.ylabel(header[yax])
    plt.title("Data and True Class Labels")
    plt.tight_layout()
    plt.show()


    # print(X)
    # return

    # for c in range(C):
    #     print(Sigma[c])

    # Min prob. of error classifier
    # Conditional likelihoods of each class given x, shape (C, N)
    class_cond_likelihoods = np.array([multivariate_normal.pdf(X[:,:11], mu[c], Sigma_diag[c]) for c in range(C)])

    # Take diag so we have (C, C) shape of priors with prior prob along diagonal
    class_priors = np.diag(priors)
    # class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
    # with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
    class_posteriors = class_priors.dot(class_cond_likelihoods)

    # MAP rule, take largest class posterior per example as your decisions matrix (N, 1)
    # Careful of indexing! Added np.ones(N) just for difference in starting from 0 in Python and labels={1,2,3}
    decisions = np.argmax(class_posteriors, axis=0) #+ np.ones(N) 

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








    return

if __name__ == '__main__':
    main()