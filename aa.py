# regularization


with np.load("tinymnist.npz") as data:
    trainData,trainTarget = data["x"],data["y"]
    validData,validTarget = data["x_valid"],data["y_valid"]
    testData,testTarget = data["x_test"],data["y_test"]

W = tf.Variable(tf.truncated_normal(shape=[64,1], stddev=0.5), name='weights')
b = tf.Variable(0.0, name='biases')
X = tf.placeholder(tf.float32, [None,64], name='input_x')

y_target = tf.placeholder(tf.float32, [None,1], name='target_y')
# Graph definition
y_predicted = tf.matmul(X,W) + b
# Error definition
meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target),
                                                 reduction_indices=1,
                                                 name='squared_error'),
                                  name='mean_squared_error')
lamda = tf.Variable(0.01, name='lamda')
penalized = tf.scalar_mul(lamda/2,tf.square(W))
# Training mechanism
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(loss=meanSquaredError+penalized)
init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
sess.run(init)
errlist = []
for i in range(1,10):
    rdData = np.arange(700)
    np.random.shuffle(rdData)
    _, err, currentW, currentb, yhat = sess.run([train, meanSquaredError+penalized, W, b, y_predicted],
                                                feed_dict={X: trainData[rdData[0:50]], y_target: trainTarget[rdData[0:50]]})
    errlist.append(err)
# Testing model
errTest = sess.run(meanSquaredError, feed_dict={X: validData, y_target: validTarget})
print("Final testing MSE: %.2f"%(errTest))
print(errlist)
# plot graph
plt.figure(1)
plt.plot(errlist)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.xlim(0,300)
plt.show()
print("Iter: %3d, MSE-train: %4.2f, weights: %s, bias: %.2f"%(i, err, currentW.T, currentb))#521-A1
