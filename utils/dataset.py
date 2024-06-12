import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import UnidentifiedImageError, Image




IMAGE_RESIZE = (128, 128, 1)

class ImageDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['general'].get('dataset_path', "./data/domain1")
        self.training_size = config['training'].get('training_size', None)
        self.validation_size = config['training'].get('validation_size', None)
        self.test_size = config['training'].get('test_size', None)
        self.normalization = config['preprocessing'].get('normalization', False)
        self.standardization = config['preprocessing'].get('standardization', False)
        self.data_augmentation = config['preprocessing'].get('data_augmentation', False)

        # Load and preprocess dataset
        self.images = self._load_images(self.dataset_path)
        self.images = self._preprocess_images(self.images)

        # Split dataset into train, validation, and test sets
        self.x_train, self.x_temp = train_test_split(self.images, test_size=self.validation_size + self.test_size)
        self.x_val, self.x_test = train_test_split(self.x_temp, test_size=self.test_size)

    def _load_images(self, folder):
        images = []
        for filename in os.listdir(folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, filename)
                img_path = os.path.normpath(img_path)
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    img = tf.image.resize(img, IMAGE_RESIZE[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=True)
                    if img.shape != IMAGE_RESIZE:
                        print(f"Shape mismatch for {img_path}: got {img.shape}, expected: {IMAGE_RESIZE}")
                    images.append(img)
                except (UnidentifiedImageError, OSError, ValueError) as e:
                    print(f"Cannot process image file {img_path}: {e}")
        return np.array(images)

    def _preprocess_images(self, images):
        images = images.astype('float32') / 255.0

        # normalization [-1, 1]
        if (self.normalization):
            images = (images - 0.5) * 2

        # standardization
        if (self.standardization):
            mean = np.mean(images, axis=(0, 1, 2))
            std = np.std(images, axis=(0, 1, 2))
            images = (images - mean) / std

        # Data Augmentation
        if (self.data_augmentation):
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)
            images = tf.image.random_brightness(images, max_delta=0.1)
            images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
            images = tf.image.random_hue(images, max_delta=0.05)
            images = tf.image.random_saturation(images, lower=0.95, upper=1.05)

        return images

    def get_train_data(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        train_dataset = train_dataset.shuffle(len(self.x_train)).batch(self.config['training']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    
    def get_validation_data(self):
        validation_dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
        validation_dataset = validation_dataset.batch(self.config['training']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return validation_dataset
    
    def get_test_data(self):
        test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        test_dataset = test_dataset.batch(self.config['training']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset