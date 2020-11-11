import tensorflow as tf


class Sampler(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sampler, self).__init__(**kwargs)

    def call(self, inputs):
        z_log_var, z_mean = inputs
        batch = z_log_var.shape[0]
        dim = z_log_var.shape[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, interm_dim, beta, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.projection_1 = tf.keras.layers.Dense(
            interm_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(beta))
        self.latent_mean = tf.keras.layers.Dense(
            latent_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(beta))
        self.latent_log_var = tf.keras.layers.Dense(
            latent_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(beta))
        self.sampler = Sampler()

    def call(self, inputs):
        x = self.projection_1(inputs)
        z_mean = self.latent_mean(x)
        z_log_var = self.latent_log_var(x)
        z = self.sampler((z_mean, z_log_var))
        return [z_log_var, z_mean, z]


class Decoder(tf.keras.Model):
    def __init__(self, original_dim, interm_dim, beta, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.layer1 = tf.keras.layers.Dense(
            interm_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(beta))
        self.layer2 = tf.keras.layers.Dense(
            original_dim,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(beta))

    def call(self, inputs):
        x = self.layer1(inputs)
        reconstruction = self.layer2(x)
        return reconstruction


class VAE(tf.keras.Model):
    def __init__(self,
                 interm_dim,
                 latent_dim,
                 original_dim,
                 beta=None,
                 **kwargs):
        super(VAE, self).__init__(*kwargs)
        self.interm_dim = interm_dim
        self.latent_dim = latent_dim
        self.original_dim = original_dim
        print('Beta recevied: ', beta)
        self.beta = beta if beta is not None else 1e-6
        print('Beta created: ', self.beta, ' of type: ', type(self.beta))
        self.encoder = Encoder(latent_dim=self.latent_dim,
                               interm_dim=self.interm_dim,
                               name='encoder',
                               beta=self.beta)
        self.decoder = Decoder(original_dim=self.original_dim,
                               interm_dim=self.interm_dim,
                               name='decoder',
                               beta=self.beta)

    def call(self, inputs, training=True):
        '''returns input reconstructuction, encoded input
    '''
        if (training is True):
            encoded_img = self.encoder(inputs)
            z_log_var, z_mean, z = encoded_img
            reconstruction = self.decoder(z)
            kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) -
                                            tf.exp(z_log_var) + 1)
            self.add_loss(kl_loss)
            return reconstruction, z
        else:
            encoded_img = self.encoder(inputs)
            _, _, z = encoded_img
            reconstruction = self.decoder(z)
            return reconstruction, z
