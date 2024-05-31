import tensorflow as tf
import numpy as np

class MNISTDataLoader:
    def __init__(self, config):
        self.config = config
        self.training_size = config['training'].get('training_size', None)
        self.validation_size = config['training'].get('validation_size', None)
        self.test_size = config['training'].get('test_size', None)
        self.img_size = config['training'].get('img_size', 28)
        self.num_classes = 10  # Number of digit classes (0-9)
        
        # Load dataset once
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_train = self.x_train[..., tf.newaxis]  # Add a channel dimension
        self.x_test = self.x_test.astype('float32') / 255.0
        self.x_test = self.x_test[..., tf.newaxis]  # Add a channel dimension
        
        # Split validation data from training data
        self.x_val = self.x_train[:self.validation_size]
        self.y_val = self.y_train[:self.validation_size]
        self.x_train = self.x_train[self.validation_size:]
        self.y_train = self.y_train[self.validation_size:]

    def _balance_dataset(self, x, y, size_per_class):
        x_balanced = []
        y_balanced = []
        for digit in range(self.num_classes):
            mask = y == digit
            x_digit = x[mask]
            y_digit = y[mask]
            x_balanced.append(x_digit[:size_per_class])
            y_balanced.append(y_digit[:size_per_class])
        
        x_balanced = np.concatenate(x_balanced)
        y_balanced = np.concatenate(y_balanced)
        
        return x_balanced, y_balanced

    def get_train_data(self):
        if self.training_size is not None:
            size_per_class = self.training_size // self.num_classes
            x_train, y_train = self._balance_dataset(self.x_train, self.y_train, size_per_class)
        else:
            x_train, y_train = self.x_train, self.y_train

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(len(y_train)).batch(self.config['training']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    
    def get_validation_data(self):
        if self.validation_size is not None:
            size_per_class = self.validation_size // self.num_classes
            x_val, y_val = self._balance_dataset(self.x_val, self.y_val, size_per_class)
        else:
            x_val, y_val = self.x_val, self.y_val
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        validation_dataset = validation_dataset.batch(self.config['training']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return validation_dataset
    
    def get_test_data(self):
        if self.test_size is not None:
            size_per_class = self.test_size // self.num_classes
            x_test, y_test = self._balance_dataset(self.x_test, self.y_test, size_per_class)
        else:
            x_test, y_test = self.x_test, self.y_test
        
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(self.config['training']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset