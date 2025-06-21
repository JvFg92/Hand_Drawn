import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image # Make sure PIL is imported for image handling

# Define the Generator Model (Example for a simple GAN)
def make_generator_model():
    model = keras.Sequential()
    # Input latent space size (e.g., 100 for a typical GAN)
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Output layer: 28x28x1 image, 'tanh' activation for output in [-1, 1]
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# Define the Discriminator Model (Example for a simple GAN)
def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # Output a single value for real/fake

    return model

# Define the GAN (for training purposes)
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return {"d_loss": disc_loss, "g_loss": gen_loss}


# Model Training Function
def train_gan_model():
    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize images to [-1, 1] for GANs
    train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    latent_dim = 100
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    gan = GAN(discriminator, generator, latent_dim)

    # Optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    gan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer)

    EPOCHS = 50 # You'll likely need more epochs for good results for a real model

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in train_dataset:
            gan.train_step(batch)
        # Add a placeholder for a progress indicator or print losses
        if (epoch + 1) % 10 == 0:
            print(f"  (Example: After Epoch {epoch+1})") # Replace with actual loss logging if you want
    print("Training complete.")

    # Save the generator weights
    # Ensure this path is where you want to save your trained model
    weights_save_path = 'trained_generator_weights.h5'
    generator.save_weights(weights_save_path)
    print(f"Generator weights saved to: {weights_save_path}")
    return generator

# Class to load and use the trained generator
class DigitGenerator:
    def __init__(self, latent_dim=100, weights_path=None):
        self.generator = make_generator_model()
        self.latent_dim = latent_dim
        self.model_loaded = False

        if weights_path:
            try:
                # Build the model by calling it once with dummy input
                # This is important before loading weights for some Keras models
                dummy_input = tf.random.normal([1, self.latent_dim])
                _ = self.generator(dummy_input)

                self.generator.load_weights(weights_path)
                self.model_loaded = True
                print(f"Generator weights loaded from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load generator weights from {weights_path}. Using untrained model. Error: {e}")
        else:
            print("No weights path provided. Using an untrained generator model.")

    def generate_image_for_app(self, target_digit=None):
        """
        Generates a single image.
        For an unconditional GAN, target_digit is not used directly in generation,
        but for a Conditional GAN (like AC-GAN or CGAN), this is where you'd
        incorporate the digit as input.
        """
        # Generate a random latent vector
        noise = tf.random.normal([1, self.latent_dim])

        # If you were using a Conditional GAN, you'd combine noise with target_digit
        # e.g., digit_one_hot = tf.one_hot([target_digit], depth=10)
        #       conditioned_noise = tf.concat([noise, digit_one_hot], axis=1)
        #       generated_image = self.generator(conditioned_noise, training=False)

        generated_image = self.generator(noise, training=False)
        generated_image = generated_image[0, :, :, 0] # Remove batch and channel dim

        # Rescale to [0, 255] and convert to uint8
        generated_image = (generated_image * 127.5 + 127.5).numpy().astype(np.uint8)

        # Convert to PIL Image for saving
        pil_image = Image.fromarray(generated_image)
        return pil_image

if __name__ == '__main__':
    # This block will run when you execute your_model_file.py directly
    print("--- Training Example ---")
    trained_generator_model = train_gan_model() # This will save weights to 'trained_generator_weights.h5'
    print("\n--- Generation Examples ---")

    # Example 1: Generate using the newly trained model
    print("Generating image using trained model...")
    trained_gen_instance = DigitGenerator(weights_path='trained_generator_weights.h5')
    generated_pil_image_trained = trained_gen_instance.generate_image_for_app(target_digit=5)
    generated_pil_image_trained.save("example_generated_digit_trained.png")
    print("Example trained image saved: example_generated_digit_trained.png")

    # Example 2: Generate using an untrained model (no weights path)
    print("\nGenerating image using untrained model (should be random noise)...")
    untrained_gen_instance = DigitGenerator() # No weights_path provided
    generated_pil_image_untrained = untrained_gen_instance.generate_image_for_app(target_digit=0)
    generated_pil_image_untrained.save("example_generated_digit_untrained.png")
    print("Example untrained image saved: example_generated_digit_untrained.png")