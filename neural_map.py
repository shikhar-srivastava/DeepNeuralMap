# Project: Deep Neural Map
# Formatting: yapf
'''Neural Map definition:

Structure: ND array of SOM neurons
Size:
Weights of each neuron: [w_j1, w_j2, .. , w_jm] for all j neurons in the SOM Map, and m is the embedding space dimensionality

best_match:
    return arg min || Z - w_j ||
        Z: embedding vector for each observation
'''

# Define Euclidian function to get distance
# Convolutional Autoencoder:
# Encoder e_theta : X -> Z
# Decoder d_phi : Z -> X'

import tensorflow as tf
import collections
import numpy as np
# For test script imports
import matplotlib.pyplot as plt
import time
import cProfile
from tensorflow.python.keras.backend import dtype
from datetime import datetime
from packaging import version
from PIL import Image
#from vae import VAE
'''
# SOM Memory with Policy, Weight tuples.
neuron = collections.namedtuple('neuron', ('weight','policy') )
'''


class NeuralMap(tf.keras.Model):
    '''
    '''
    def __init__(self,
                 shape,
                 embed_dim,
                 alpha,
                 eta_0=None,
                 sigma_0=None,
                 dtypes=tf.float16,
                 map_init=None,
                 **kwargs):
        '''

            Inputs:
                    shape: Shape/Structure of the map
                    embed_dim: Embedding shape/ Weight shape of each neuron in SOM
                    alpha: Hyperparameter controlling rate of Neighborhood shrinking (& learning rate reduction) [Rule of thumb: alpha ~= No. of iterations/5]
                    eta_0: SOM Learning rate constant in eta_n = eta_0 * exp(n/alpha). 
                    sigma_0: Hyperparameter controlling maximum size of neighborhood region [Usually set to max(map shape) / 2]
                    map_init: Map initialization value for the map of shape [embed_dim]. 
          

        '''
        super(NeuralMap, self).__init__(**kwargs)

        self.shape = shape  # Defines shape of neurons in SOM L = [l1,l2,...ln]
        self.embed_dim = embed_dim  # Defines weight dimensionality of neurons: m
        self.dtypes = dtypes
        self.map_shape = self.shape + self.embed_dim
        self.alpha = alpha  #Rule of thumb: alpha ~= No. of iterations/5
        self.eta_0 = eta_0 if eta_0 is not None else 0.3
        self.sigma_0 = sigma_0 if sigma_0 is not None else max(
            self.shape) / 2.0
        self.label_map = np.zeros(self.shape, dtype=np.object)
        self.initializer = 'random_uniform'  #tf.keras.initializers.RandomUniform(minval=0,maxval=255)

        # Utility vars

        # Computational Cache for some operations
        self.distances = None  # Euclidian distances computed for current batch
        self.grid = self._index_grid(self.shape)
        self._grid = self.grid  # Specific shape of self.grid is needed for Gradient computation. [Recomputing this takes time]
        self.grid = tf.reshape(self.grid, [-1, self.grid.shape[-1]])

        # Initialize step
        self.step = tf.constant(
            0,
            dtype=tf.float16)  #float16 for type consistency for all constants.

        # Create SOM weight array
        print('Creating SOM of shape %s with weight dimension %s' %
              (self.shape, self.embed_dim))
        if (map_init is None):
            self.SOM = self.add_weight('deep_neural_map',
                                       shape=self.map_shape,
                                       initializer=self.initializer,
                                       dtype=self.dtypes,
                                       trainable=True)
        else:
            assert self.embed_dim == map_init.shape, 'NeuralMap/_init_:Incorrect shape of map initializer : Shape(map_init)  =/ [Shape(memory_shape) + shape(embed_dim)]'
            self.SOM = tf.Variable(tf.ones(self.map_shape) * (tf.reshape(
                map_init,
                list(tf.ones([len(self.shape)], dtype=tf.int32).numpy()) +
                self.embed_dim)),
                                   dtype=tf.float16)
        # SOM Shape: [l1,l2,..,ln,M]. Therefore, SOM - Z gives shape [l1,l2,...,ln,M] with Ni - Z in each element.

    @tf.function
    def eta(self, n):
        return self.eta_0 * tf.math.exp(-n / self.alpha)

    @tf.function
    def sigma(self, n):
        return self.sigma_0 * tf.math.exp(-n / self.alpha)

    @tf.function
    def H(self, index_diff, n):
        ''' u: tuple (dim1, dim2, ..., dimN),
            j: tuple(dim1, dim2, ..., dimN)
        '''
        return tf.math.exp((-1 * (tf.square(tf.norm(index_diff, axis=-1)))) /
                           (2 * tf.math.square(self.sigma(n))))

    def read(self, Z):
        '''Take embedding Z, and read the neuron in the SOM most representative of that encoding. With ('weight','policy') tuple, this will read the corressponding policy weights as well.
        '''
        return NotImplementedError()

    @tf.function
    def best_match_neuron(self, Z, weights=False):
        '''
            Function that returns the best_match_neurons U.
            Inputs:
                Z: embedding inputs of shape [batch_size, self.embed_dim]
                weights: (boolean) True: returns weights of the Best_match_neurons
        '''
        return self._best_match_neuron(Z, weights)[0]

    @tf.function
    def _best_match_neuron(self, Z, weights=False):
        '''
            Inputs:
                Z: embedding inputs of shape [batch_size, self.embed_dim]
                weights: (boolean) True: returns weights of the Best_match_neurons
            In order to reuse computation, best_match_neuron will return:
            Returns:
                U, diff, distances
        '''
        # embeddings = tf.random.normal([batch_size, embed_dim], mean = 0.0, stddev = 1.0, dtype= tf.float16, seed = 10, name = 'embedding_sample') # embeddings random input
        # Reshape Z
        assert Z.shape[1:] == self.embed_dim, (
            str(self.__class__.__name__) +
            ': Embeddings Z do not have -> shape Z.shape[1:] = embedding_dimension'
        )
        Z_shape = tf.concat([[tf.shape(Z)[0]],
                             tf.ones(len(self.shape), dtype=tf.int32),
                             tf.constant(self.embed_dim)],
                            axis=0)
        Z_ = tf.reshape(
            Z, shape=Z_shape
        )  # Reshape embeddings batch to allow distance computation with N-dim SOM
        diff = tf.math.subtract(Z_, self.SOM, name='best_match_neuron/diff')
        distances = self.nd_norm(
            diff
        )  # N-dimensional L2 Norm to get distance |tf.norm(Z_ - self.SOM, ord='euclidean', axis = -1)
        # Flatten self.distances to [batch_size, shape[0]*shape[1]*...*shape[n-1]]
        # Convert first to [batch_size, 1] then get nd_index
        distance_shape = tf.concat([
            tf.expand_dims(tf.shape(distances)[0], 0),
            tf.constant(-1, shape=(1, ))
        ],
                                   axis=0)
        U = self.nd_index(
            tf.argmin(tf.reshape(distances, shape=distance_shape),
                      axis=-1))  #converts indices to N-dim indices
        if (weights):
            return tf.gather_nd(self.SOM, U), diff, distances
        else:
            return U, diff  #, distances

    @tf.function
    def call(self, inputs, step=None, iter=1, training=True):
        '''
        Inputs: Batch of embeddings of shape: [batch_size, embed_dim]
            Returns:
                if(training):
                    accumulate kl_loss with self.add_loss()
                    return Batch of neuron-weights 'u' closest to input embeddings. Shape: [batch_size, embed_dim]
                else:
                    return Batch of neuron-weights 'u' closest to input embeddings. Shape: [batch_size, embed_dim]
        '''
        if (training is False):
            # -- Inference only
            return self.best_match_neuron(inputs, weights=True)
        else:
            # -- Training
            # Read & update memory
            if (step is None):
                step = self.step
            # Define iteration loop using tf.while
            i = tf.constant(0, dtype=tf.float16)

            gradient = None
            while (tf.less(i, tf.constant(iter, dtype=tf.float16))):
                gradient = self.compute_gradient(inputs, step)
                self.update_memory(
                    gradient)  # Updates memory and increments step
                i = tf.add(i, 1)
                # Add loss to model
            kl_loss = self.get_kl_loss(inputs)

            return {'kl_loss': kl_loss, 'delta': gradient}

    def assign_labels(self, types, Z):
        ''' Z: shape - [batch_size, embed_dims]
            types: shape - [batch_size]
        '''
        U = self.best_match_neuron(Z)
        for i in range(U.shape[0]):
            self.label_map[tuple(U[i])] = types[i]

    @tf.function
    def compute_gradient(self, Z, n):
        '''
        Calculates the delta of change in the SOM by topological spread of a 'write' onto the selected neuron with spread to neighborhood neurons.
        We do:
            delta = η(n) [ H(u,j)(n)(Zi − wj(n)) ]
            wj(n + 1) = wj(n) + delta
            H(u,j)(n) : neighborhood function
            η(n) : learning rate
        '''
        U, diff = self._best_match_neuron(Z)
        #indices of best match neurons in order of Z batch | [batch_size, len(shape)]
        #U_weights = tf.gather_nd(self.SOM, U) # best match neuron weights in order of Z batch | [batch_size, embed_dim]

        U_reshape = tf.concat([
            tf.expand_dims(tf.shape(U)[0], 0),
            tf.ones([len(self.shape)], dtype=tf.int32),
            tf.expand_dims(tf.shape(U)[1], 0)
        ],
                              axis=0)
        U_ = tf.reshape(U, shape=U_reshape)
        index_diff = tf.cast(U_ - self._grid, dtype=tf.float16)
        # [batch_size, [shape], len(shape)]

        h = self.H(index_diff, n)
        h_reshape = tf.concat(
            [tf.shape(h),
             tf.ones([len(self.embed_dim)], dtype=tf.int32)],
            axis=0)
        h = tf.reshape(h, shape=h_reshape)
        return self.eta(n) * h * diff

    @tf.function
    def density_q(self, Z):
        '''
        '''
        raw_q = 1 / (1 + self.nd_norm(
            tf.expand_dims(Z, axis=1) - tf.reshape(self.SOM,
                                                   ([-1] + self.embed_dim))))
        return raw_q / (tf.reduce_sum(raw_q, axis=-1, keepdims=True)
                        )  # raw_p will always be [batch_size, l]

    @tf.function
    def density_p(self, q):
        '''
            density_p is the Target Distribution p(W|Z) that our SOM must train towards.
            Why? It essentially prioritizes each neuron by how rarely it is 'selected' in the batch of embeddings. This means that separation of clusters is improved.
            Larger the batch size in calculating density_p or the more representative the batch is of the population of the embeddings, the better the learning of the SOM.
            @TODO: Makes unncessary extra call to self.density_q. pass q as param.
        '''
        # since shape of 'q' is guaranteed to be [batch_size, l] irrespective of SOM lattice structure, therefore we can use axis = 0/1 in below reductions
        raw_p = tf.square(q) / (tf.reduce_sum(q, axis=0, keepdims=True))
        return raw_p / (tf.reduce_sum(raw_p, axis=1, keepdims=True))

    @tf.function
    def kl_divergence(self, p, q):
        '''Inputs:
                    p: [batch_size, l]
                    q: [batch_size, l]
                            where l (total neurons in SOM lattice) = self.shape[0] * self.shape[1] * ... * self.shape[tf.rank(self.SOM)]
            Returns:
                    KLD(p||q) = SUM_p(p * log(p/q)) : [batch_size,]
        '''
        return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)

    @tf.function
    def get_kl_loss(self, Z):
        q = self.density_q(Z)
        p = self.density_p(q)
        return self.kl_divergence(p, q)

    @tf.function
    def update_memory(self, delta, type='average'):
        ''' Idea is to update the memory after batch of deltas are received. With (neuron,policy) memory tuple, update_memory would store the policy to the memory tuples as well.
        Inputs:
            delta: [batch_size, self.shape]
            In equation, wj(n + 1) = wj(n) + η(n) H(u,j)(n)(Zi − wj(n))
            we take delta = [ η(n) H(u,j)(n)(Zi − wj(n)) ] computed for batch of Z.
                # For a batch, conventionally in SOMs (Kohonen 1990), we take mean of delta across batch axis for each wj neuron.

         * Increment step count
        '''
        if (type == 'sum'):
            self.SOM.assign_add(tf.reduce_sum(
                delta, axis=0))  # axis=0 for reducing along batch
        else:
            # type = 'average'
            self.SOM.assign_add(tf.reduce_mean(
                delta, axis=0))  # axis=0 for reducing along batch
        # Update steps
        self.step = tf.add(self.step, 1)

    @tf.function
    def nd_index(self, U_):
        '''Convert indices for flattened tensor to corresponding indices for ND tensor
      '''
        return tf.gather(self.grid, U_)

    @tf.function
    def nd_norm(self, diff):
        '''
            Norm of the last len(embed_dim) axes. 
        
            Input: Tensor of shape ( [dim0....dimnN] + [embed_dim] ) [[shape] + [embed_dim]]
            Returns: L2 Normed Tensor of shape ( [dim0....dimN] )
            @optimized as: Flatten the last embed_dim dimensions and then taken regular tf.norm
        '''
        axes = np.arange(start=-1, stop=-1 * len(self.embed_dim) - 1, step=-1)
        return tf.math.reduce_euclidean_norm(diff, axis=axes)

    def _index_grid(self, shape):
        x = np.indices(shape)
        x = np.reshape(x, [len(shape), -1, 1])
        x = np.concatenate(x, axis=-1)
        x = np.reshape(x, shape + [len(shape)])
        return x


@tf.function
def tensor_to_grid(x):
    '''
        Utility for Visualization of SOM - From SOM tensor, generate 2d stitch of all neurons into one 2D tensor
        Input:
            (SOM Tensor) x: Shape [X,Y,D1,D2,D3,D4]
        Returns:
            (2D Stich Tensor) :  [X,Y]
    '''
    rows = tf.unstack(x, axis=0)
    concat_across_x = [tf.concat(tf.unstack(x, axis=0), axis=1) for x in rows]
    return tf.concat(concat_across_x, axis=0)


@tf.function
def som_to_grid(neural_map):
    '''
        VIZ Utility: Converts SOM tensor into [1,X,Y,1] shape tensor by stitching neurons in 2D map to a 2D tensor of shape [X,Y]
    '''
    img_size = int(np.sqrt(neural_map.embed_dim))
    grid = tensor_to_grid(
        tf.reshape(neural_map.SOM,
                   shape=[
                       neural_map.shape[0], neural_map.shape[1], img_size,
                       img_size
                   ]))
    return tf.reshape(grid,
                      shape=tf.concat(
                          [tf.constant([1]),
                           tf.shape(grid),
                           tf.constant([1])],
                          axis=0))


def trainer(iterations, epochs, shape, batch_size, max_steps):

    # Set up logging and Tracing.
    #tf.profiler.experimental.start('logs/neural_map_v1/profile')
    # Fetch Dataset
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = tf.cast(tf.reshape(x_train, (x_train.shape[0], -1)),
                      dtype=tf.float16) / 255
    # Feed unsupervised average to initialize
    #data_avg = tf.reduce_mean(x_train, axis=0)
    # Call only one tf.function when tracing.
    #shape = [5, 5]

    neural_map = NeuralMap(
        shape=shape,
        embed_dim=[784],
        dtype=tf.float16,
        alpha=(1.0 / 5.0 *
               max_steps),  # how quickly the neighborhood region shrinks.
        sigma_0=None,  # max size of the neighborhood region 
        eta_0=0.3,
        map_init=None)
    # Data
    train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(
        buffer_size=2 * batch_size).batch(batch_size)
    # Metrics

    kl_loss = tf.keras.metrics.Mean()
    distortion = tf.keras.metrics.Mean()
    # Logging material here
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/neural_map_' + (str(shape[0]) + 'x' +
                                   str(shape[1])) + '/%s' % stamp
    #tb_callback = tf.keras.callbacks.TensorBoard(logdir)
    #tb_callback.set_model(neural_map)
    #tf.summary.trace_on(graph=True, profiler=True)

    with tf.summary.create_file_writer(logdir).as_default() as writer:
        step = 0
        for epoch in range(0, epochs):
            kl_loss.reset_states()
            distortion.reset_states()
            for batch in enumerate(train_ds):
                losses = neural_map(batch[1],
                                    step=step,
                                    iter=iterations,
                                    training=True)
                kl_loss.update_state(tf.reduce_mean(losses['kl_loss']))
                distortion.update_state(tf.reduce_mean(losses['delta']))

            tf.summary.image('Epoch : ' + str(epoch),
                             data=som_to_grid(neural_map),
                             max_outputs=3,
                             step=step)
            #tf.summary.scalar('step', step, step=step)
            #tf.summary.scalar('epoch', epoch, step=epoch)
            tf.summary.scalar('kl_loss', kl_loss.result(), step=step)
            tf.summary.scalar('distortion', distortion.result(), step=step)

            tf.print('========= Epoch ', step, '======= KL_Loss:',
                     kl_loss.result(), 'Distortion:', distortion.result())
            '''tf.summary.trace_export(name="map",
                                    step=0,
                                    profiler_outdir=logdir)'''
            step += 1
    #tf.profiler.experimental.stop()
    '''#Viz
    print('Visualizing')
    neural_map.viz('full_' + str(shape[0]) + 'x' + str(shape[1]))'''


if (__name__ == '__main__'):
    #cProfile.run('run()', '/home/dm1/shikhar/deep_neural_maps/stats_optim_3')
    iterations = 10
    epochs = 8
    batch_size = 50
    max_steps = 8
    shapes = [[2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [75, 75],
              [100, 100]]
    for shape in shapes:
        trainer(iterations, epochs, shape, batch_size, max_steps)
