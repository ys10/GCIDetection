import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# Launch a graph in a sesion
sess = tf.Session()
# Evaluate a tensor c
result = sess.run(c)
print(result)