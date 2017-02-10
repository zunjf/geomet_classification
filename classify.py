import tensorflow as tf
import sys
import numpy
from myimg import getImage

# number of classes
nClass = 2

# simple model (boolean)
simpleModel = False

# dimensions of images
height = 32
width = 32

# associate the "label" and "image" objects with the corresponding features read from
# a single example in the training data file
label, image = getImage("data/train-00000-of-00001")

# and similarity for the validation data
vlabel, vimage = getImage("data/validation-00000-of-00001")

# associate the "label_batch" and "image_batch" objects with a randomly selected batch
# of labels and images respectively
imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label], batch_size=100,
    capacity=2000,
    min_after_dequeue=1000
)

# and smilarly for the validation data
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], batch_size=100,
    capacity=2000,
    min_after_dequeue=1000
)

# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

# x is the input array, which will contain the data from an image
# this creates a placeholder for x, to be populated later
x = tf.placeholder(tf.float32, [None, width*height])

# similarly, we have a placeholder for true outputs (obtained from labels)
y_ = tf.placeholder(tf.float32, [None, nClass])

if simpleModel:
    # run simple model y = Wx + b in Tensorflow "MNIST" tutorial
    print "Running Simple Model.."

    # initialize weights and biases to zero
    # W maps input to output so is of size: (number of pixels) * (Number of classes)
    W = tf.Variable(tf.zeros([width*height, nClass]))

    # b is vector which has a size corresponding to number of class
    b = tf.Variable(tf.zeros([nClass]))

    # define output calc (for each class) y = softmax(Wx+b)
    # softmax gives probability distribution across all classes
    y = tf.nn.softmax(tf.matmul(x, W)+b)
else:
    # run CNN model given in "Expert MNIST" Tensorflow tutorial

    # functions to init small positif and biases
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # set up "vanilla" version of convolution and pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print "Running CNN Model.."
    nFeatures1 = 32
    nFeatures2 = 64
    nNeurosfc = 1024

    # use functions to init weights and biases
    # nFeatures1 features for each patch of size 5x5
    # SAME weights used for all patches
    # 1 input channel
    W_conv1 = weight_variable([5, 5, 1, nFeatures1])
    b_conv1 = bias_variable([nFeatures1])

    # reshape raw image data to 4D tensor. 2nd and 3rd indexes are W, H,
    # fourth means 1 colour channel/pixel
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    x_image = tf.reshape(x, [-1, width, height, 1])

    # hidden layer 1
    # pool(convolution(Wx) + b)
    # pool reduces each dim by factor of 2
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # similarly for second layer, with nFeatures2 features per 5x5 patch
    # input is nFeatures1 (number of features output from previous layer)
    W_conv2 = weight_variable([5, 5, nFeatures1, nFeatures2])
    b_conv2 = bias_variable([nFeatures2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # denseley connected layer. Similar to above, but operating
    # on entire image (rather than patch) which has been reduced by a factor of 4
    # in each dimension
    # so use large number of neurons

    # check our dimensions are a multiple of 4
    if width % 4 or height % 4:
        print 'Error: modulo 4'
        sys.exit(1)

    W_fc1 = weight_variable([(width/4) * (height/4) * nFeatures2, nNeurosfc])
    b_fc1 = bias_variable([nNeurosfc])

    # flatten output from previous layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, (width/4) * (height/4) * nFeatures2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # reduce overfitting by applying dropout
    # each neuron is kept with probability keep_prob
    keep_prop = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)

    # create readout layer which outputs to nClass categories
    W_fc2 = weight_variable([nNeurosfc, nClass])
    b_fc2 = bias_variable([nClass])

    # define output calc (for each class) y = softmax(Wx+b)
    # softmax gives probability distribution across all classes
    # this is not run until later
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# measure of error of our model
# this needs to be minimised by adjusting W and b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# get mean of all entries in corrent prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
sess.run(tf.initialize_all_variables())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# start training
nSteps = 1000
for i in range(nSteps):
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

    # run the training step with feed of images
    if simpleModel:
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    else:
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prop: 0.5})

    if(i+1)%100 == 0:
        # perform validation
        # get validation batch
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])

        if simpleModel:
            train_accuracy = accuracy.eval(feed_dict={x: vbatch_xs, y_:vbatch_ys})
        else:
            train_accuracy = accuracy.eval(feed_dict={
                x: vbatch_xs, y_:vbatch_ys, keep_prop: 1.0})

        print ("step %d, training accuracy %g" %(i+1, train_accuracy))

coord.request_stop()
coord.join(threads)

