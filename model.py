import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

tf.compat.v1.disable_eager_execution()

l2_lambda = 1e-04


def _residual_block(x, size, dropout=False, dropout_prob=0.5, seed=None):
    residual = tf.compat.v1.layers.batch_normalization(
        x
    )  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.compat.v1.layers.conv2d(
        residual,
        filters=size,
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
    )
    if dropout:
        residual = tf.nn.dropout(residual, 1 - (dropout_prob), seed=seed)
    residual = tf.compat.v1.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.compat.v1.layers.conv2d(
        residual,
        filters=size,
        kernel_size=3,
        padding="same",
        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
    )
    if dropout:
        residual = tf.nn.dropout(residual, 1 - (dropout_prob), seed=seed)

    return residual


def one_residual(x, keep_prob=0.5, seed=None):
    nn = tf.compat.v1.layers.conv2d(
        x,
        filters=32,
        kernel_size=5,
        strides=2,
        padding="same",
        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
    )
    nn = tf.compat.v1.layers.max_pooling2d(nn, pool_size=3, strides=2)

    rb_1 = _residual_block(nn, 32, dropout_prob=keep_prob, seed=seed)

    nn = tf.compat.v1.layers.conv2d(
        nn,
        filters=32,
        kernel_size=1,
        strides=2,
        padding="same",
        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
    )
    nn = tf.keras.layers.add([rb_1, nn])

    nn = tf.compat.v1.layers.flatten(nn)

    return nn


class TfInference:
    def __init__(self, observation_shape, action_shape, graph_location, seed=1234):
        # model definition
        self._observation = None
        self._action = None
        self._computation_graph = None
        self._optimizer = None

        self.tf_session = tf.compat.v1.InteractiveSession()

        # restoring
        self.tf_checkpoint = None
        self.tf_saver = None

        self.seed = seed

        self._initialize(observation_shape, action_shape, graph_location)

    def predict(self, state):
        action = self.tf_session.run(
            self._computation_graph,
            feed_dict={
                self._observation: [state],
            },
        )
        return np.squeeze(action)

    def computation_graph(self):
        model = one_residual(self._preprocessed_state, seed=self.seed)
        model = tf.compat.v1.layers.dense(
            model,
            units=64,
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal"),
                seed=self.seed,
            ),
            bias_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal"),
                seed=self.seed,
            ),
        )
        model = tf.compat.v1.layers.dense(
            model,
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal"),
                seed=self.seed,
            ),
            bias_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal"),
                seed=self.seed,
            ),
        )

        model = tf.compat.v1.layers.dense(model, self._action.shape[1])

        model = tf.tanh(model)

        return model

    def _initialize(self, input_shape, action_shape, storage_location):
        if not self._computation_graph:
            self._create(input_shape, action_shape)
            self._storing(storage_location)

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize(frame, (60, 80)), self._observation)
        and_standardize = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resize)
        self._preprocessed_state = and_standardize

    def _create(self, input_shape, output_shape):
        self._observation = tf.compat.v1.placeholder(dtype=tf.float32, shape=input_shape, name="state")
        self._action = tf.compat.v1.placeholder(dtype=tf.float32, shape=output_shape, name="action")
        self._pre_process()

        self._computation_graph = self.computation_graph()

    def _storing(self, location):
        self.tf_saver = tf.compat.v1.train.Saver()

        self.tf_checkpoint = tf.train.latest_checkpoint(location)
        if self.tf_checkpoint:
            self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
        else:
            raise IOError("No model found...")

    def close(self):
        self.tf_session.close()
