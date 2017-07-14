import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 1
n_chain = 20

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
hrand = tf.placeholder("float", [None, n_hidden_1])
vrand = tf.placeholder("float", [None, n_input])

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
    
def free_energy(v):
    ''' Function to compute the free energy '''
    wx_b = tf.add(tf.matmul(v, W) ,bh)
    vbias_term = tf.reshape(tf.matmul(v, tf.reshape(bv,(-1,1))),(-1,))
    hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)), axis=1)
    return -hidden_term - vbias_term


def do_chain():
    h_prob,h_samples = get_h_given_x(W,bh,x)
    v_prob,v_samples = get_x_given_h(W,bv,h_prob)
    for i in xrange(n_chain):
        h_prob,h_samples = get_h_given_x(W,bh,v_prob)
        v_prob,v_samples = get_x_given_h(W,bv,h_prob)
    return v_samples


v_samples = do_chain()

cost = tf.reduce_mean(free_energy(x))- tf.reduce_mean(free_energy(v_samples))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.stop_gradient(v_samples) 
reconst_error = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(x, v_samples)),axis=1))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    #=================
    # TRAINING CYCLES
    #=================
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_rec = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = (batch_x > 0.0).astype('float')
            # Run optimization for contrastive divergance
            _,vrec,rec,c = sess.run([ optimizer,v_samples,reconst_error,cost], feed_dict={x: batch_x,
                                             hrand: np.random.rand(batch_size,n_hidden_1),
                                             vrand: np.random.rand(batch_size,n_input)
                                             })
            
            # Compute average loss
            avg_cost += c / total_batch
            avg_rec += rec / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost),
                  "Rec=", "{:.9f}".format(avg_rec)
                  )
                      
    print("Optimization Finished!")
    
    #=================
    #   TEST MODEL
    #=================
    avg_cost = 0.
    avg_rec = 0.0
    total_batch = int(mnist.test.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        batch_x = (batch_x > 0.0).astype('float')
        # Run optimization for contrastive divergance
        vrec,rec,c = sess.run([v_samples,reconst_error,cost], feed_dict={x: batch_x,
                                         hrand: np.random.rand(batch_size,n_hidden_1),
                                         vrand: np.random.rand(batch_size,n_input)
                                         })
        avg_cost += c / total_batch
        avg_rec += rec / total_batch
        
    print( "Test avg cost=", "{:.9f}".format(avg_cost),
          "Test avg Rec=", "{:.9f}".format(avg_rec)
          )   
    
    #==================
    # SAMPLE FROM MODEL
    #==================
    # generate images with 1000 gibbs sampling from the model distribution
    n_chain = 1000
    batch_x, batch_y = mnist.test.next_batch(batch_size)
    batch_x = (batch_x > 0.0).astype('float')
    # Run optimization for contrastive divergance
    vrec,rec,c = sess.run([v_samples,reconst_error,cost], feed_dict={x: batch_x,
                            hrand: np.random.rand(batch_size,n_hidden_1),
                            vrand: np.random.rand(batch_size,n_input)
                            })    
    for ii in xrange(64):
        plt.subplot(8,8,ii+1)
        plt.imshow(vrec[ii,:].reshape((28,28)),cmap='gray')     
    plt.show()