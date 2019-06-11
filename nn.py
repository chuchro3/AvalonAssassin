from data_parser import parseData
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.0008
num_steps = 500000
batch_size = 128
display_step = 10000

# Network Parameters
n_hidden_1 = 16 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons
n_hidden_3 = 8 # 3rd layer number of neurons
#num_input = 5*5*5*5 # feature size
num_input = 5*7 # feature size
num_classes = 5 # 5 players

def main():
    
    #get data from source
    data, res, merlins, mostCorrects, percivals, vts, _ = parseData()
    num_train = int(len(merlins) * .9)
    data_train = np.array(data[:num_train])
    data_test = np.array(data[num_train:])
    merlins_test = merlins[num_train:]
    res_test = res[num_train:]
    X_train = data_train.reshape( (data_train.shape[0], -1) )
    Y_train = np.array(merlins[:num_train])
    Y_train = one_hot_encode(Y_train)
    X_test = data_test.reshape( (data_test.shape[0], -1) )
    Y_test = np.array(merlins_test)
    Y_test = one_hot_encode(Y_test)
    print(Y_test.shape)
    
    #tf data
    #tfdata_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    #iterator = tfdata_train.make_one_shot_iterator()
    #next_element = iterator.get_next()
    
    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct model
    logits = neural_net(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    accs = []
    ts = []
    losses = []
    print("Starting training")
    print("hidden 1 size: " + str(n_hidden_1))
    print("hidden 2 size: " + str(n_hidden_2))
    print("hidden 3 size: " + str(n_hidden_3))
    print("batch size: " + str(batch_size))
    print("iterations: " + str(num_steps))
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps+1):
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            ridx = np.random.randint(num_train, size=batch_size)
            batch_x = X_train[ridx]
            batch_y = Y_train[ridx]
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

                accs.append(acc)
                ts.append(step)
                losses.append(loss)

        print("Optimization Finished!")

        # Calculate accuracy for test data
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: X_test,
                                          Y: Y_test}))

        plt.figure("Training Accuracy")
        plt.xlabel("Time Step")
        plt.ylabel("Batch Accuracy")
        #plt.ylim(0, 1)
        plt.plot(ts, accs)
        plt.savefig("training_acc", bbox_inches="tight")
        
        plt.figure("Training Loss")
        plt.xlabel("Time Step")
        plt.ylabel("Batch Loss")
        plt.plot(ts, losses)
        plt.savefig("training_loss", bbox_inches="tight")

# Create model
def neural_net(x, weights, biases):
    # Hidden fully connected layer 
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer 
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer 
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

def one_hot_encode(y):
    new_y = np.zeros( (y.shape[0], num_classes) )
    for i in range(y.shape[0]):
        new_y[i][y[i]] = 1
    return new_y

 
if __name__ == "__main__":
    main()
