import tensorflow as tf

class SobolevNetwork:
    def __init__(self, input_dim, num_hidden):
        self.input_dim = input_dim 
        self.num_hidden = num_hidden
        self.W1 = tf.Variable(tf.random_normal([self.input_dim, self.num_hidden],
                                   stddev=0.1))
        self.b1 = tf.Variable(tf.ones([self.num_hidden]))
        self.W2 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],
                                   stddev=0.1))
        self.b2 = tf.Variable(tf.ones([self.num_hidden]))
        self.W3 = tf.Variable(tf.random_normal([self.num_hidden, 1],
                                   stddev=0.1))
        self.b3 = tf.Variable(tf.ones([1]))
        self.weights = [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3)]

    def forward(self, X):
        out = X
        for W, b in self.weights:
            out = tf.nn.relu(tf.matmul(out, W) + b)
        return out