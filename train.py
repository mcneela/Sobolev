import tensorflow as tf
import numpy as np
import data
from model import SobolevNetwork

INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_SAMPLES = 1024
NUM_HIDDEN = 256
NUM_EPOCHS = 10

X = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
y = tf.placeholder(tf.float32, shape=[None])
y_der = tf.placeholder(tf.float32, shape=[None])

model = SobolevNetwork(INPUT_DIM, NUM_HIDDEN)

y_p = model.forward(X)
dy = tf.gradients(y_p, X)

loss = tf.reduce_mean(tf.pow(y_p - y, 2) + tf.pow(dy - y_der, 2))
optim = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch_num, epoch in enumerate(range(NUM_EPOCHS)):
    batch_samples = data.genTrainData(d=INPUT_DIM, num_samples=NUM_SAMPLES)
    X_train = [s[0] for s in batch_samples]
    y_train = [s[1] for s in batch_samples]
    dy_train = [s[2] for s in batch_samples]
    train_dict = {X: X_train, y: y_train, y_der: dy_train}
    _, curr_loss, y_preds, dy_preds = sess.run([optim, loss,
                                          y_p, dy], feed_dict=train_dict)
    print("Epoch: %d, Loss: %f" % (epoch_num, curr_loss))

