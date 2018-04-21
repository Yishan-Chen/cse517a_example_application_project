import sys
sys.path.insert(0, '/Users/op')

import numpy as np
import scipy.spatial.distance as sp
import preprocessing

# compute covariance matrix for GP with different kernel:
def compute_se_kernel_matrix(X, Y, lsquared):
    if X is Y: dist = sp.squareform(sp.pdist(X, "sqeuclidean"))
    else: dist = sp.cdist(X, Y, "sqeuclidean")
    K = np.exp(-dist / (2.0*lsquared))
    return K

# def compute_se_kernel_matrix(Xs, Ys, lsquared):
#     return np.dot(Xs, np.transpose(Ys))


def gp(X, T, X_s, lsquared):

    K = compute_se_kernel_matrix(X, X, lsquared)
    K_s_T = compute_se_kernel_matrix(X_s, X, lsquared)
    K_inv = np.linalg.inv(K)
    K_s_T_times_K_inv = np.dot(K_s_T, K_inv)

    size, classes = len(X_s), T.max()+1
    predictions = np.zeros((size, classes))

    for k in range(classes):
        k_class_vs_rest = np.where(T == k, 1.0, -1.0)
        result = np.dot(K_s_T_times_K_inv, k_class_vs_rest)
        predictions[:, k] = result

    labels = np.argmax(predictions, axis=1)

    return labels


# error evaluation
def error_evaluate(T_predicted, T_true):
    count = np.count_nonzero(np.array(T_predicted != T_true))
    return count * 100.0 / len(T_predicted)


if __name__ == "__main__":

    numTrain = 1000
    numTest =  600
    isDeskew = True
    isNormalize = True
    lsquared = 1.0

    print "Reading data..."
    # data pre-processing
    X_train, T_train = preprocessing.get_training_data(numTrain, normalize=isNormalize, deskew=isDeskew)
    X_test, T_test = preprocessing.get_testing_data(numTest, normalize=isNormalize, deskew=isDeskew)
    print "{0} training data read".format(len(X_train))
    print "{0} testing data read".format(len(X_test))

    # running a Gaussian process on training and testing sets
    T_predicted = gp(X_train, T_train, X_test, lsquared=1)

    # computing the error
    print "Testing Set Error: {0:.3f}".format(
        error_evaluate(T_predicted, T_test)
    )