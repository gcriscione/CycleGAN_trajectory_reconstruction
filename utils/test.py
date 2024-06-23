import tensorflow as tf
from utils import plot_images
from config.config_file import config

def test_model(generator, noise_adder, test_data, logger):
    total_loss = 0
    batch_count = 0

    for i, images in enumerate(test_data):
        noisy_images = noise_adder.add_noise(images)

        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=-1)
        if len(noisy_images.shape) == 3:
            noisy_images = tf.expand_dims(noisy_images, axis=-1)

        reconstructed_images = generator(noisy_images, training=False)
        loss_cycle = tf.reduce_mean(tf.abs(images - reconstructed_images))

        total_loss += loss_cycle
        batch_count += 1

        # Plot images for testing
        if i % 50 == 0:
            plot_images(images, noisy_images, reconstructed_images, min(5, images.shape[0]), config['general']['show_plots'], config['general']['save_plots'], None, f'test[{i}]', config['training']['seed'])
        
    average_loss = total_loss / batch_count
    logger.info(f"Test Loss: {average_loss}")
    print(f"Test Loss: {average_loss}")
    return average_loss
