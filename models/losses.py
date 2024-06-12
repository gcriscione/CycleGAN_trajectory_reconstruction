import tensorflow as tf

class Losses:
    def __init__(self, config):
        loss_type = config['training']['loss_function']
        self.lambda_cycle = config['training']['lambda_cycle']
        self.lambda_identity = config['training'].get('lambda_identity', 0.5 * self.lambda_cycle)
        
        if loss_type == 'MSE':
            self.adversarial_loss = tf.keras.losses.MeanSquaredError()
        elif loss_type == 'BCE':
            self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def generator_loss(self, fake_output):
        return self.adversarial_loss(tf.ones_like(fake_output), fake_output)

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
        return self.lambda_cycle * self.cycle_loss_fn(real_image, reconstructed_image)
    
    def identity_loss(self, real_image, same_image):
        if len(real_image.shape) != 4:
            real_image = tf.expand_dims(real_image, axis=0)
        if len(same_image.shape) != 4:
            same_image = tf.expand_dims(same_image, axis=0)

        assert len(real_image.shape) == 4, f"real_image deve avere 4 dimensioni, ma ha {len(real_image.shape)}"
        assert len(same_image.shape) == 4, f"same_image deve avere 4 dimensioni, ma ha {len(same_image.shape)}"

        if real_image.shape[1:3] != same_image.shape[1:3]:
            real_image = tf.image.resize(real_image, same_image.shape[1:3])
        return self.lambda_identity * self.identity_loss_fn(real_image, same_image)

    def line_segment_loss(self, real_image, generated_image):
        if real_image.shape != generated_image.shape:
            print(f"Shape mismatch: real_image shape {real_image.shape}, generated_image shape {generated_image.shape}")
            generated_image = tf.image.resize(generated_image, real_image.shape[1:3])
        real_image_lines = tf.where(real_image < -0.5, tf.ones_like(real_image), tf.zeros_like(real_image))
        generated_image_lines = tf.where(generated_image < -0.5, tf.ones_like(generated_image), tf.zeros_like(generated_image))
        return tf.reduce_mean(tf.abs(real_image_lines - generated_image_lines))