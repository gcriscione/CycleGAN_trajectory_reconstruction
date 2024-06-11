import tensorflow as tf
import numpy as np
import random

class NoiseAdder:
    def __init__(self, config):
        self.noise_type = config["noise_adder"]["noise_type"]
        self.salt_pepper_ratio = config["noise_adder"]["salt_pepper_ratio"]
        self.gaussian_mean = config["noise_adder"]["gaussian_mean"]
        self.gaussian_std = config["noise_adder"]["gaussian_std"]
        self.line_segment_params = config["noise_adder"].get("line_segment_params", {
            "line_count": 5,
            "line_thickness": 2,
        })

        # Dictionary of noise functions
        self.noise_functions = {
            "salt_and_pepper": self._add_salt_and_pepper_noise,
            "gaussian": self._add_gaussian_noise,
            "speckle": self._add_speckle_noise,
            "poisson": self._add_poisson_noise,
            "line_segments": self._add_line_segments
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
    
    # Add line segments to the images.
    def _add_line_segments(self, images):
        images = images.numpy()
        noisy_images = images.copy()

        line_count = self.line_segment_params["line_count"]
        line_thickness = self.line_segment_params["line_thickness"]

        for img in noisy_images:
            for _ in range(line_count):
                x1, y1 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
                x2, y2 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
                img = self._draw_line(img, (x1, y1), (x2, y2), (0, 0, 0, 1), line_thickness)

        return tf.convert_to_tensor(noisy_images)
    
    # Helper function to draw a line on an image.
    def _draw_line(self, img, start, end, color, thickness):
        img = ((img + 1) * 127.5).astype(np.uint8).copy()
        
        x1, y1 = start
        x2, y2 = end

        length = int(np.hypot(x2 - x1, y2 - y1))
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length

        for i in range(length):
            x = int(x1 + dx * i)
            y = int(y1 + dy * i)
            for t in range(-thickness // 2, thickness // 2):
                for s in range(-thickness // 2, thickness // 2):
                    if 0 <= x + t < img.shape[1] and 0 <= y + s < img.shape[0]:
                        img[y + s, x + t] = color

        img = (img / 127.5) - 1
        return img
