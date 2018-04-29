from keras.datasets import mnist
from sklearn import metrics
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

clf = GaussianNB()
lr = LogisticRegression()

# load the MNIST dataset
(X_train_ori, Y_train_ori), (x_test, y_test) = mnist.load_data()

# Data preprocessing
lbl = preprocessing.LabelEncoder()

#
iter = 0
scores = []
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits)

for train_index, test_index in skf.split(X_train_ori, Y_train_ori):
    iter += 1
    print('fold {}'.format(iter), 'Training') #, 'train', train_index, 'test', test_index)
    print('Testing')
    x_train, x_validation = X_train_ori[train_index], X_train_ori[test_index]
    y_train, y_validation = Y_train_ori[train_index], Y_train_ori[test_index]

    # transform each image from a 28 by 28 pixel matrix to a 784 pixel vector
    pixel_count = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
    x_validation = x_validation.reshape(x_validation.shape[0], pixel_count).astype('float32')
    
    # normalize inputs from gray scale of 0-255 to values between 0-1
    x_train = x_train / 255
    x_validation = x_validation / 255
    
    # PCA
    PCs = 30 # num of PCs
    pca = PCA(n_components=PCs)
    pca.fit(x_train)
    x_train = pca.fit_transform(x_train)
    x_validation = pca.fit_transform(x_validation)
    acc_vars = pca.explained_variance_ratio_

    # Dummy Variable
    Y_train = pd.get_dummies(y_train)
    Y_validation = pd.get_dummies(y_validation)

    # reshape result
    x_test1 = x_test
    x_test1 = x_test1.reshape(x_test1.shape[0], pixel_count).astype('float32')
    x_test1 = x_test1 / 255
    x_test1 = pca.fit_transform(x_test1)
    Y_test = np_utils.to_categorical(y_test, 10)

    # Prediction
    model = lr.fit(x_train, y_train)
    pred = model.predict(x_test1)
    scores.append(metrics.accuracy_score(y_test, pred))
    #print(iter, x_test.shape, x_test1.shape)

print('scores for each fold:', scores)
print('avg score:{:.4f}'.format(sum(scores)/n_splits))

acc_sum = []
acc_va = 0
for va in acc_vars:
    acc_va += va
    acc_sum.append(acc_va)
len_acc = range(len(acc_sum))
plt.plot(len_acc, acc_sum)
