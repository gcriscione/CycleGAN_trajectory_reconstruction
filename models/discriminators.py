import io
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras import layers
import logging

# Set up the logger
logging.basicConfig(filename='result/logs/models.log', level=logging.INFO, filemode='w')
logger = logging.getLogger(__name__)

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.match_dims = None
        self.filters = filters

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.match_dims = tf.keras.layers.Conv2D(self.filters, 1, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.match_dims:
            inputs = self.match_dims(inputs)
        
        x += inputs
        return self.relu(x)

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        # Set seed for reproducibility
        seed = config['training'].get('seed', -1)
        if seed is not None and seed != -1:
            tf.random.set_seed(seed)

        # Create the discriminator model
        self.model = tf.keras.Sequential()
        for layer_config in config["discriminator"]["layers"]:
            if layer_config["type"] == "residual_block":
                self.add_residual_blocks(layer_config)
            else:
                self.add_layer(layer_config)

    def add_layer(self, layer_config):
        if layer_config["type"] == "conv":
            self.model.add(layers.Conv2D(
                filters=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                strides=layer_config["stride"],
                padding='same'))
            logger.info(f"Added Conv2D layer with {layer_config['out_channels']} filters")
        elif layer_config["type"] == "conv_transpose":
            self.model.add(layers.Conv2DTranspose(
                filters=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                strides=layer_config["stride"],
                padding='same'))
            logger.info(f"Added Conv2DTranspose layer with {layer_config['out_channels']} filters")
        else:
            logger.warning(f"Unknown layer type: {layer_config['type']}")

        if "activation" in layer_config:
            if layer_config["activation"] == "LeakyReLU":
                self.model.add(layers.LeakyReLU())
                logger.info("Added LeakyReLU activation")
            elif layer_config["activation"] == "Sigmoid":
                self.model.add(layers.Activation('sigmoid'))
                logger.info("Added Sigmoid activation")

        if "batch_norm" in layer_config and layer_config["batch_norm"]:
            self.model.add(layers.BatchNormalization())
            logger.info("Added BatchNormalization")

        if "dropout" in layer_config:
            self.model.add(layers.Dropout(layer_config["dropout"]))
            logger.info(f"Added Dropout with rate {layer_config['dropout']}")

    def add_residual_blocks(self, layer_config):
        num_blocks = layer_config.get("num_blocks", 1)
        filters = layer_config["out_channels"]
        kernel_size = layer_config["kernel_size"]

        for _ in range(num_blocks):
            self.model.add(ResNetBlock(filters=filters, kernel_size=kernel_size))
            logger.info(f"Added ResNetBlock with {filters} filters")

    def call(self, inputs, training=None):
        return self.model(inputs)
    
    def __str__(self):
        self.model.build(input_shape=(None, 28, 28, 1))
        with io.StringIO() as buf, redirect_stdout(buf):
            self.model.summary()
            model_summary = buf.getvalue()
        return model_summary