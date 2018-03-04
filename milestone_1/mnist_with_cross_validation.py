import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create graph
sess = tf.Session()

# Load the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

for n in range(0, 55000, 5500):
    # 10 fold cross validation - get training data
    train_images = np.concatenate((mnist.train.images[n:55000, ], mnist.train.images[0:n, ]), axis=0)
    train_labels = np.concatenate((mnist.train.labels[n:55000, ], mnist.train.labels[0:n, ]), axis=0)

    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, 784], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 10], dtype=tf.float32)

    # Create variables for linear regression
    W = tf.Variable(tf.zeros(shape=[784, 10]))
    b = tf.Variable(tf.zeros(shape=[10]))

    # Declare model operations
    model_output = tf.nn.softmax(tf.add(tf.matmul(x_data, W), b))

    # Declare loss function (L2 loss)
    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(model_output), reduction_indices=[1]))

    # Declare optimizer
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy_loss)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    # for i in range(1):
    #     idx_images = np.random.choice(train_images.shape[0], size=50000, replace=False)
    #     idx_labels = np.random.choice(train_labels.shape[0], size=50000, replace=False)
    #     new_train_images = train_images[idx_images, :]
    #     new_train_labels = train_labels[idx_labels, :]
    sess.run(train_step, feed_dict={x_data: train_images, y_target: train_labels})

    # 10 fold cross validation - get validation data
    validation_images = mnist.train.images[n:n+5500, ]
    validation_labels = mnist.train.labels[n:n+5500, ]

    # Evaluating
    correct_prediction = tf.equal(tf.argmax(model_output, 1), tf.argmax(y_target, 1))

    # this accuracy returns the mean value of an array of 1s and 0s.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # retrun the accuracy on the test set.
    acc = np.ndarray([])
    print(type(sess.run(accuracy, feed_dict={x_data: validation_images, y_target: validation_labels})))
    print("Accuracy: ", sess.run(accuracy, feed_dict={x_data: validation_images, y_target: validation_labels}))

sess.close()
