import tensorflow as tf
from utils import plot_images
from config.config_file import config

def validate_model(epoch, generator1, generator2, noise_adder, val_loader, losses, logger):
    total_loss_G = 0
    batch_count = 0

    for i, (images, _) in enumerate(val_loader):
        noisy_images = noise_adder.add_noise(images)

        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=-1)
        if len(noisy_images.shape) == 3:
            noisy_images = tf.expand_dims(noisy_images, axis=-1)

        reconstructed_images_G1 = generator1(images, training=False)
        reconstructed_images_G2 = generator2(noisy_images, training=False)

        loss_cycle_G1 = losses.cycle_consistency_loss(images, reconstructed_images_G1)
        loss_cycle_G2 = losses.cycle_consistency_loss(noisy_images, reconstructed_images_G2)

        total_loss_G += loss_cycle_G1 + loss_cycle_G2
        batch_count += 1

        # Plot images for validation
        if i % 50 == 0:
            plot_images(images, noisy_images, reconstructed_images_G1, min(5, images.shape[0]), config['general']['show_plots'], config['general']['save_plots'], (epoch+1), f'validate[{i}]', config['training']['seed'])

    average_loss = total_loss_G / batch_count
    logger.info(f"Validation Loss - Epoch: {epoch+1}, Average Loss: {average_loss}")

    print(f"Validation Loss - Epoch: {epoch+1}, Average Loss: {average_loss}")
    return average_loss
