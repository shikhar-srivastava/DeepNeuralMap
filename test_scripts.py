    
    # Sample Test for SOM
    print('Test NeuralMap class with sample execution')
    neural_map = NeuralMap(shape=[10, 10], embed_dim=[100], dtype=tf.float32)
    print(neural_map)
    #neural_map(tf.ones([1]))
    #print(neural_map.shape)
    z = tf.random.normal([1000, 100],
                         mean=0.0,
                         stddev=1.0,
                         dtype=tf.float32,
                         seed=10,
                         name='embedding_sample')  # embeddings random input
    #print(neural_map.read(z))
    print('SOM Looks like:  ', neural_map.SOM)
    print(neural_map(z, iter=1, training=True))
    print('SOM after updating: ', neural_map.SOM)

    print('Test NeuralMap class with sample execution')
    #Time Test Sequence: MNIST
    
    neural_map = NeuralMap(shape=[30, 30], embed_dim=[784], dtype=tf.float32)
    print('#1: ', neural_map)
    print('#2 SOM Looks like:  ', neural_map.SOM)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.reshape(x_train, [x_train.shape[0], -1])
    x_train = tf.cast(x_train, dtype=tf.float32)
    times = []
    lens = []
    for i in range(1, 20000, 5000):
        start_time = time.time()
        print(neural_map(x_train[:i], iter=i, training=True))
        elapsed = (time.time() - start_time)
        times.append(elapsed)
        lens.append(i)
        print("--- %s seconds ---" % (elapsed))
    plt.plot(lens, times)
    plt.show()