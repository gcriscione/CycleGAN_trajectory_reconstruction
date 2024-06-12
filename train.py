import tensorflow as tf
import logging
import json
from models import Generator, Discriminator, Losses
from utils import ImageDataLoader, NoiseAdder, plot_images, stats, create_checkpoint_manager, validate_model, test_model
from config.config_file import config

CONFIGURATION_FILE = "result/setup/config.json"
MODEL_FILE = "result/setup/models.txt"

# Configure logger for training
training_logger = logging.getLogger('TrainingLogger')
training_logger.setLevel(logging.INFO)
handler = logging.FileHandler('result/logs/training.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
training_logger.addHandler(handler)

# Enable memory growth for GPUs
def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# Initialize models, optimizers, and other components
def initialize_components(config):
    G1 = Generator(config)
    G2 = Generator(config)
    D1 = Discriminator(config)
    D2 = Discriminator(config)

    optimizer_G1 = tf.keras.optimizers.Adam(
        learning_rate=config['training']['generator_learning_rate'], 
        beta_1=config['training']['beta1'], 
        beta_2=config['training']['beta2']
    )
    optimizer_G2 = tf.keras.optimizers.Adam(
        learning_rate=config['training']['generator_learning_rate'], 
        beta_1=config['training']['beta1'], 
        beta_2=config['training']['beta2']
    )
    optimizer_D1 = tf.keras.optimizers.Adam(
        learning_rate=config['training']['discriminator_learning_rate'], 
        beta_1=config['training']['beta1'], 
        beta_2=config['training']['beta2']
    )
    optimizer_D2 = tf.keras.optimizers.Adam(
        learning_rate=config['training']['discriminator_learning_rate'], 
        beta_1=config['training']['beta1'], 
        beta_2=config['training']['beta2']
    )

    noise_adder = NoiseAdder(config)
    losses = Losses(config)

    return G1, G2, D1, D2, optimizer_G1, optimizer_G2, optimizer_D1, optimizer_D2, noise_adder, losses

# Perform training for one epoch
def train_one_epoch(epoch, train_loader, G1, G2, D1, D2, optimizer_G1, optimizer_G2, optimizer_D1, optimizer_D2, noise_adder, losses):
    for i, images in enumerate(train_loader):
        noisy_images = noise_adder.add_noise(images)

        # Ensure images have 4 dimensions
        if len(images.shape) == 3:
            images = tf.expand_dims(images, axis=-1)
        if len(noisy_images.shape) == 3:
            noisy_images = tf.expand_dims(noisy_images, axis=-1)

        with tf.GradientTape() as tape_D1, tf.GradientTape() as tape_D2, tf.GradientTape() as tape_G1, tf.GradientTape() as tape_G2:
            fake_images_G2 = G2(noisy_images, training=True)
            fake_images_G1 = G1(images, training=True)

            real_output_D1 = D1(images, training=True)
            fake_output_D1 = D1(fake_images_G2, training=True)

            real_output_D2 = D2(noisy_images, training=True)
            fake_output_D2 = D2(fake_images_G1, training=True)

            loss_D1 = losses.discriminator_loss(real_output_D1, fake_output_D1)
            loss_D2 = losses.discriminator_loss(real_output_D2, fake_output_D2)

            loss_G1 = losses.generator_loss(fake_output_D2)
            loss_G2 = losses.generator_loss(fake_output_D1)

            reconstructed_images_G1 = G1(fake_images_G2, training=True)
            reconstructed_images_G2 = G2(fake_images_G1, training=True)

            loss_cycle_G1 = losses.cycle_consistency_loss(images, reconstructed_images_G1)
            loss_cycle_G2 = losses.cycle_consistency_loss(noisy_images, reconstructed_images_G2)

            # Identity loss
            identity_loss_G1 = losses.identity_loss(images, G1(images, training=True))
            identity_loss_G2 = losses.identity_loss(noisy_images, G2(noisy_images, training=True))

            # Line segment loss
            line_segment_loss_G1 = losses.line_segment_loss(images, fake_images_G2)
            line_segment_loss_G2 = losses.line_segment_loss(noisy_images, fake_images_G1)

            total_loss_G1 = loss_G1 + loss_cycle_G1 + identity_loss_G1 + line_segment_loss_G1
            total_loss_G2 = loss_G2 + loss_cycle_G2 + identity_loss_G2 + line_segment_loss_G2

        gradients_D1 = tape_D1.gradient(loss_D1, D1.trainable_variables)
        gradients_D2 = tape_D2.gradient(loss_D2, D2.trainable_variables)
        gradients_G1 = tape_G1.gradient(total_loss_G1, G1.trainable_variables)
        gradients_G2 = tape_G2.gradient(total_loss_G2, G2.trainable_variables)

        optimizer_D1.apply_gradients(zip(gradients_D1, D1.trainable_variables))
        optimizer_D2.apply_gradients(zip(gradients_D2, D2.trainable_variables))
        optimizer_G1.apply_gradients(zip(gradients_G1, G1.trainable_variables))
        optimizer_G2.apply_gradients(zip(gradients_G2, G2.trainable_variables))

        training_logger.info(stats((epoch+1), (i+1), loss_D1, loss_D2, total_loss_G1 + total_loss_G2))
        if (i + 1) % 10 == 0 or (i == 0):
            print(stats((epoch+1), (i+1), loss_D1, loss_D2, total_loss_G1 + total_loss_G2))
        if (i + 1) % 500 == 0:
            plot_images(images, noisy_images, reconstructed_images_G1, min(5, images.shape[0]), config['general']['show_plots'], config['general']['save_plots'], (epoch+1), f'training[{i+1}]', config['training']['seed'])

# Main training loop
def train():
    enable_memory_growth()
    
    data_loader = ImageDataLoader(config)
    train_loader = data_loader.get_train_data()
    val_loader = data_loader.get_validation_data()
    test_loader = data_loader.get_test_data()

    G1, G2, D1, D2, optimizer_G1, optimizer_G2, optimizer_D1, optimizer_D2, noise_adder, losses = initialize_components(config)

    model = {'generator1': G1, 'generator2': G2, 'discriminator1': D1, 'discriminator2': D2}
    optimizer = {'generator1': optimizer_G1, 'generator2': optimizer_G2, 'discriminator1': optimizer_D1, 'discriminator2': optimizer_D2}

    checkpoint, checkpoint_manager = create_checkpoint_manager(model, optimizer)

    # Restore latest checkpoint if it exists
    mode = config['general']['mode']
    if mode == 'train':
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            try:
                with open(CONFIGURATION_FILE, 'w') as json_file:
                    json.dump(config, json_file, indent=4)
                print(f"Configurazione salvata in {CONFIGURATION_FILE}")
            except Exception as e:
                print(f"Errore nel salvataggio dell'oggetto in {CONFIGURATION_FILE}: {e}")
    
            try:
                with open(MODEL_FILE, 'w') as file:
                    file.write(f"\t\tGENERATOR 1\n{G1}\n\n")
                    file.write(f"\t\tGENERATOR 2\n{G2}\n\n")
                    file.write(f"\t\tDISCRIMINATOR 1\n{D1}\n\n")
                    file.write(f"\t\tDISCRIMINATOR 2\n{D2}\n\n")
                print(f"Configurazione modelli salvata in {MODEL_FILE}")
            except Exception as e:
                print(f"Errore nel salvataggio in {MODEL_FILE}: {e}")

            print("Starting from scratch.")

        training_logger.info("TRAINING:")
        print("\n\t\tTRAINING:")
        for epoch in range(config['training']['num_epochs']):
            train_one_epoch(epoch, train_loader, G1, G2, D1, D2, optimizer_G1, optimizer_G2, optimizer_D1, optimizer_D2, noise_adder, losses)

            validate_model(epoch, G1, G2, noise_adder, val_loader, losses, training_logger)

            # Save checkpoint at the end of each epoch
            checkpoint_manager.save()
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Test the model after training
        test_model(G1, noise_adder, test_loader, training_logger)

    elif mode == 'test':
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint} for testing.")
            test_model(G1, noise_adder, test_loader)
        else:
            print("No checkpoint found. Cannot test the model.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CycleGAN Image Denoising')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run: train or test')
    args = parser.parse_args()

    train()