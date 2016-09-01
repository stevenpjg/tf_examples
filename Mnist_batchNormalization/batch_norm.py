import tensorflow as tf
decay = 0.999
epsilon = 1e-3
class batch_norm:
    def __init__(self,inputs,is_training,sess,bn_param=None):
        self.sess = sess        
        self.scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        self.beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        self.pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        self.pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0])        
        
        self.train_mean = tf.assign(self.pop_mean,
                               self.pop_mean * decay + self.batch_mean * (1 - decay))  
        self.train_var = tf.assign(self.pop_var,
                              self.pop_var * decay + self.batch_var * (1 - decay))
        
        def training(): return tf.nn.batch_normalization(inputs,
                self.batch_mean, self.batch_var, self.beta, self.scale, epsilon)
    
    
        def testing(): return tf.nn.batch_normalization(inputs,
            self.pop_mean, self.pop_var, self.beta, self.scale, epsilon)
            
        self.bnorm = tf.cond(is_training,training,testing) 
        