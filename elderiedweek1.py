import tensorflow as tf
import numpy as np

h = tf.constant("Hello")
w = tf.constant(" World!")
hw=h+w
with tf.Session() as sess: ans = sess.run(hw)
print (ans)

x = tf.constant([1,2,3],name='x',dtype=tf.float32)
print(x.dtype)
x=tf.cast(x,tf.int64)
print(x.dtype)



sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0, 5)
print("The content of 'c': \n {}\n".format(c.eval()))
sess.close()

a = tf.constant([[ 1,2,3], [4,5,6]])

print(a.get_shape())