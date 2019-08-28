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

x = tf.constant([1,0,1])
print(x.get_shape())

x = tf.expand_dims(x,1)
print(x.get_shape())

b = tf.matmul(a,x)

sess = tf.InteractiveSession()
print("matmul result: \n {}".format(b.eval()))
sess.close()

with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    c2 = tf.constant(4, dtype=tf.int32, name='c')

print(c1.name)
print(c2.name)

y_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

print(y_data,w_data)