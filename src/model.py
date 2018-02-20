import tensorflow as tf

class SobolevNetwork:
    def __init__(self, input_dim, num_hidden):
        self.input_dim = input_dim 
        self.num_hidden = num_hidden
        self.W1 = tf.Variable(tf.random_normal([self.input_dim, self.num_hidden],
                                  stddev=0.1))
        self.W2 = tf.Variable(tf.random_normal([self.num_hidden, self.num_hidden],
                                   stddev=0.1))
        self.W3 = tf.Variable(tf.random_normal([self.num_hidden, 1],
                                   stddev=0.1))
        self.weights = [self.W1, self.W2, self.W3]

    def forward(self, X):
        out = X
        for W in self.weights[:-1]:
            out = tf.nn.relu(tf.matmul(out, W))
        out = tf.matmul(out, self.weights[-1])
        return out