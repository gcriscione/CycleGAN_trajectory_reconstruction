import tensorflow as tf
import numpy as np

class NoiseAdder:
    def __init__(self, config):
        self.noise_type = config["noise_adder"]["noise_type"]
        self.salt_pepper_ratio = config["noise_adder"]["salt_pepper_ratio"]
        self.gaussian_mean = config["noise_adder"]["gaussian_mean"]
        self.gaussian_std = config["noise_adder"]["gaussian_std"]

        # Dictionary of noise functions
        self.noise_functions = {
            "salt_and_pepper": self._add_salt_and_pepper_noise,
            "gaussian": self._add_gaussian_noise,
            "speckle": self._add_speckle_noise,
            "poisson": self._add_poisson_noise
        }

    # Add noise to the input images based on the specified noise type.
    def add_noise(self, images):
        if self.noise_type in self.noise_functions:
            return self.noise_functions[self.noise_type](images)
        else:
            raise ValueError("Unsupported noise type")

    # Add salt and pepper noise to the images.
    def _add_salt_and_pepper_noise(self, images):
        images = images.numpy()  # Convert to numpy array
        noisy_images = images.copy()
        num_salt = np.ceil(self.salt_pepper_ratio * images.size * 0.5)
        num_pepper = np.ceil(self.salt_pepper_ratio * images.size * 0.5)

        # Add Salt noise
        coords = [np.random.randint(0, i, int(num_salt)) for i in images.shape]
        noisy_images[coords[0], coords[1], coords[2]] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i, int(num_pepper)) for i in images.shape]
        noisy_images[coords[0], coords[1], coords[2]] = 0

        return tf.convert_to_tensor(noisy_images)  # Convert back to tensor

    # Add Gaussian noise to the images
    def _add_gaussian_noise(self, images):
        mean = self.gaussian_mean
        std = self.gaussian_std
        gaussian_noise = np.random.normal(mean, std, images.shape)
        noisy_images = images + gaussian_noise
        return tf.convert_to_tensor(np.clip(noisy_images, -1.0, 1.0))
    
    # Add speckle noise to the images.
    def _add_speckle_noise(self, images):
        noise = np.random.randn(*images.shape)
        noisy_images = images + images * noise
        return tf.convert_to_tensor(np.clip(noisy_images, -1.0, 1.0))

    # Add Poisson noise to the images.
    def _add_poisson_noise(self, images):
        noisy_images = np.random.poisson(images * 255.0) / 255.0
        return tf.convert_to_tensor(np.clip(noisy_images, -1.0, 1.0))