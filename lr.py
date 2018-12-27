import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
learning_parameter =0.01
epoch = 300
samples=50
x_train=np.linspace(0,20,samples)
y_train=6*x_train+7*np.random.randn(samples) 
#linear regression is to fit a line on to a noisy dataset 
Y=tf.placeholder(tf.float32)
X=tf.placeholder(tf.float32)
w=tf.Variable(np.random.randn(),name='weight')
b=tf.Variable(np.random.randn(),name='bias')
pred=X*w+b
cost=tf.reduce_sum((pred-Y)**2)/samples
optimizer=tf.train.GradientDescentOptimizer(learning_parameter).minimize(cost)
#initializing the variables that we have defined
init=tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  
  for i in range(epoch):
    for x, y in zip(x_train,y_train):
      sess.run(optimizer,feed_dict = {X : x , Y : y})
      w1=sess.run(w)
      b1=sess.run(b)
    #print(w1,"  ",b1)
    cost_iter=sess.run(cost,feed_dict = { X : x , Y : y })
    print('epoch',i,'cost',cost_iter,'w',w1,'b',b1)
  weight=sess.run(w)
  bias=sess.run(b)
plt.plot(x_train,weight*x_train+bias)
plt.plot(x_train,y_train,'o')
plt.plot(x_train,y_train)
plt.show()
#sess.close()

