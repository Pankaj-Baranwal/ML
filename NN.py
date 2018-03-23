import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one_hot converts inputs into a vector of zeroes with one entry as 1.
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

'''
What is the difference between tf.Variable and tf.placeholder?
In short, you use tf.Variable for trainable variables such as weights (W) and biases (B) for your model.
tf.placeholder is used to feed actual training examples.
So, Variables are trained over time, placeholders are input data that doesn't change as your model trains (like input images, and class labels for those images).
Placeholders need not be assigned an init value while Variables need to be.
'''
# [height, width]
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# tf.random_normal produces matrix of random normals (numbers) of the specified dimension
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	# tf.matmul = matrix multiplication
	# tf.multiply = element-wise multiplication
	l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
	
	# RELU = Re(ctified) L(inear) (U)nit
	# RelU = activation function = f(x) = max(0,x)
	# Adds non-linearity. It is one of the simplest non-linear functions. So, backpropagation is very easy to implement.
	l1 = tf.nn.relu(l1)
	l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
	l2 = tf.nn.relu(l2)
	l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)

	'''
	StackOverflow:
	tf.nn.softmax produces just the result of applying the softmax function to an input tensor. 
	The softmax "squishes" the inputs so that sum(input) = 1; it's a way of normalizing. 
	The shape of output of a softmax is the same as the input - it just normalizes the values. 
	The outputs of softmax can be interpreted as probabilities.

	In contrast, tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result 
	after applying the softmax function. So, it brings linearity to output.
	The cross entropy is a summary metric.
	It sums across the elements. The output of tf.nn.softmax_cross_entropy_with_logits on a shape [2,5] tensor is of shape [2,1] 
	(the first dimension is treated as the batch)

	If you want to do optimization to minimize the cross entropy, AND you're softmaxing after your last layer, 
	you should use tf.nn.softmax_cross_entropy_with_logits

	Tensorflow:
	It computes softmax cross entropy bw logits and labels.
	It measures the probability error in discrete classification tasks in which the classes are mutually exclusive.
	So, each input will have one and only one label.

	NOTE: Do not call this op with output of softmax as it performs a softmax on logits internally for efficieny. It might produces incorrect results.
	logits = unscaled log probabilities.
	'''
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# AdamOptimzer already has init learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# epoch_x = training input data, epoch_y = training input label
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print ('Epoch', epoch, 'Completed out of ', hm_epochs, ' loss: ', epoch_loss)

		# tf.argmax will return the index of the max value
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		# changing the data type from tensor to float
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print ('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)