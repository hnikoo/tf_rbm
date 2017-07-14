import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
hrand = tf.placeholder("float", [None, n_hidden_1])
vrand = tf.placeholder("float", [None, n_input])
batchSize = tf.placeholder("float")

W = tf.Variable(0.01*tf.random_normal([n_input, n_hidden_1]))
bh = tf.Variable(0.01*tf.random_normal([n_hidden_1]))
bv = tf.Variable(0.01*tf.random_normal([n_input]))

DW = tf.Variable(tf.zeros([n_input, n_hidden_1]))
Dbh = tf.Variable(tf.zeros([n_hidden_1]))
Dbv = tf.Variable(tf.zeros([n_input]))

def sample_prob(probs, rand):
    return tf.nn.relu(tf.sign(probs - rand))


def get_h_given_x(W,bh,x): 
    h_prob = tf.nn.sigmoid(tf.add(tf.matmul(x, W), bh))
    h_samples = sample_prob(h_prob,hrand)
    return h_prob,h_samples

def get_x_given_h(W,bv,h): 
    v_prob = tf.nn.sigmoid(tf.add(tf.matmul(h,tf.transpose(W)), bv))
    v_samples = sample_prob(v_prob,vrand)
    return v_prob,v_samples


def get_deltaW(v,h,hprime,vprime):
    positive = tf.matmul(tf.transpose(v),h)
    negative = tf.matmul(tf.transpose(vprime),hprime)
    dW = tf.multiply(tf.subtract(positive ,negative),1 / batchSize)
    dbv = tf.reduce_mean(tf.subtract(v,vprime),axis=0)
    dbh = tf.reduce_mean(tf.subtract(h,hprime),axis=0)
    return dW,dbv,dbh


def updataW(W,bh,bv,DW,Dbh,Dbv,dW,dbv,dbh):
    DW = 0.5*DW + 0.1*dW -0.00002*W
    Dbh = 0.5*Dbh + 0.1*dbh -0.00002*bh
    Dbv = 0.5*Dbv + 0.1*dbv -0.00002*bv
    W = tf.add(W ,DW) 
    bh = tf.add(bh, Dbh) 
    bv = tf.add(bv ,Dbv)
    return W,bh,bv,DW,Dbh,Dbv
    
def free_energy(v):
    ''' Function to compute the free energy '''
    wx_b = tf.add(tf.matmul(v, W) ,bh)
    vbias_term = tf.matmul(v, tf.reshape(bv,(-1,1)))
    hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)), axis=1)
    return -hidden_term - vbias_term



h_prob,h_samples = get_h_given_x(W,bh,x)
v_prob,v_samples = get_x_given_h(W,bv,h_samples) # change h_prob to h_samples
h_prime_prob,h_prime_samples = get_h_given_x(W,bh,v_prob)
v_prime_prob,v_prime_samples = get_x_given_h(W,bv,h_prime_prob)
dW,dbv,dbh = get_deltaW(x,h_prob,h_prime_prob,v_prime_prob)
W,bh,bv,DW,Dbh,Dbv = updataW(W,bh,bv,DW,Dbh,Dbv,dW,dbv,dbh)
cost = tf.reduce_mean(free_energy(x) - free_energy(v_prime_prob))
reconst_error = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(x, v_prob)),axis=1))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization for contrastive divergance
            hsam,ddw,rec,c = sess.run([ v_samples,dW,reconst_error,cost], feed_dict={x: batch_x,
                                             hrand: np.random.rand(batch_size,n_hidden_1),
                                             vrand: np.random.rand(batch_size,n_input),
                                             batchSize: batch_size
                                             })
            
            print  rec
            plt.subplot(2,1,1);plt.imshow(hsam[0,:].reshape((28,28)),cmap='gray')
            plt.subplot(2,1,2);plt.imshow(ddw,cmap='gray')
            plt.pause(0.001)
            plt.clf()
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            #for ii in xrange(9):
                #plt.subplot(3,3,ii+1)
                #plt.imshow(vrec[ii,:].reshape((28,28)),cmap='gray')
            #plt.pause(0.001)
            #plt.clf()            
    print("Optimization Finished!")

    # Test model

