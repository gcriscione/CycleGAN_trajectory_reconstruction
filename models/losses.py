import tensorflow as tf

class Losses:
    def __init__(self, config):
        loss_type = config['training']['loss_function']
        self.lambda_cycle = config['training']['lambda_cycle']
        
        if loss_type == 'MSE':
            self.adversarial_loss = tf.keras.losses.MeanSquaredError()
        elif loss_type == 'BCE':
            self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.cycle_loss = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def generator_loss(self, fake_output):
        adversarial_loss = self.adversarial_loss(tf.ones_like(fake_output), fake_output)
        mse_loss = tf.keras.losses.MeanSquaredError()(tf.ones_like(fake_output), fake_output)
        return adversarial_loss + mse_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.adversarial_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.adversarial_loss(tf.zeros_like(fake_output), fake_output)
        return (real_loss + fake_loss) / 2

    def cycle_consistency_loss(self, real_image, reconstructed_image):
        # Assicurati che le dimensioni corrispondano e che siano 4D
        if len(real_image.shape) != 4:
            real_image = tf.expand_dims(real_image, axis=0)
        if len(reconstructed_image.shape) != 4:
            reconstructed_image = tf.expand_dims(reconstructed_image, axis=0)

        # Aggiungi un controllo per garantire che entrambe le immagini siano 4D
        assert len(real_image.shape) == 4, f"real_image deve avere 4 dimensioni, ma ha {len(real_image.shape)}"
        assert len(reconstructed_image.shape) == 4, f"reconstructed_image deve avere 4 dimensioni, ma ha {len(reconstructed_image.shape)}"

        if real_image.shape[1:3] != reconstructed_image.shape[1:3]:
            real_image = tf.image.resize(real_image, reconstructed_image.shape[1:3])
        return self.lambda_cycle * self.cycle_loss(real_image, reconstructed_image)
    
    def identity_loss(self, real_image, same_image):
        if len(real_image.shape) != 4:
            real_image = tf.expand_dims(real_image, axis=0)
        if len(same_image.shape) != 4:
            same_image = tf.expand_dims(same_image, axis=0)

        assert len(real_image.shape) == 4, f"real_image deve avere 4 dimensioni, ma ha {len(real_image.shape)}"
        assert len(same_image.shape) == 4, f"same_image deve avere 4 dimensioni, ma ha {len(same_image.shape)}"

        if real_image.shape[1:3] != same_image.shape[1:3]:
            real_image = tf.image.resize(real_image, same_image.shape[1:3])
        return self.identity_loss_fn(real_image, same_image)