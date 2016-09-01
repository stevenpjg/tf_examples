import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from batch_norm import batch_norm
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#parameters
model_path = "/tmp/mnist_bn.ckpt"

# Generate predetermined random weights so the networks are similarly initialized
w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

# Small epsilon value for the BN transform
epsilon = 1e-3

# Placeholders
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
is_training = tf.placeholder(tf.bool, [])

# Layer 1
w1 = tf.Variable(w1_initial)
z1 = tf.matmul(x,w1)
bn1 = batch_norm(z1, is_training, sess)
l1 = tf.nn.sigmoid(bn1.bnorm)

#Layer 2
w2 = tf.Variable(w2_initial)
z2 = tf.matmul(l1,w2)
bn2 = batch_norm(z2, is_training, sess)
l2 = tf.nn.sigmoid(bn2.bnorm)

# Softmax
w3 = tf.Variable(w3_initial)
b3 = tf.Variable(tf.zeros([10]))
y  = tf.nn.softmax(tf.matmul(l2, w3))

 # Loss, Optimizer and Predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.initialize_all_variables())

#saver
saver = tf.train.Saver()
acc = []
for i in range(40000):
    batch = mnist.train.next_batch(60)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], is_training: True})
    sess.run([bn1.train_mean,bn1.train_var,bn2.train_mean,bn2.train_var],feed_dict={x: batch[0], y_: batch[1],is_training: True})
    if i % 50 is 0:
        res = sess.run([accuracy],feed_dict={x: mnist.test.images, y_: mnist.test.labels,is_training: True})
        acc.append(res[0])
        print 'Iteration No.:', i        
        print 'Accuracy with BN', res[0]
        print '\n'


#save model weights to disk:
save_path = saver.save(sess,model_path)
print "Model saved in file:%s" %save_path

predictions = []
correct = 0
for i in range(100):
    pred, corr = sess.run([tf.arg_max(y,1), accuracy],
                         feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]],is_training: False})
    correct += corr
    predictions.append(pred[0])
print("PREDICTIONS:", predictions)
print("ACCURACY:", correct/100)

