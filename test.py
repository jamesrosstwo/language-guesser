import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# params
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# i/o
x = tf.placeholder(tf.float32)  # in

linear_model = W * x + b  # model
y = tf.placeholder(tf.float32)  # out

# loss
# Y is the desired output. linear_model is the computed output. The loss describes how far from the desired result the neural network was.
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# optimization
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    File_Writer = tf.summary.FileWriter('graph', sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print(sess.run([W, b]))
