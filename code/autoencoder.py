import numpy as np
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras import layers, models


# load audio file in the player
audio_path = "audio/bassloop.wav"

# 1. load audio file
signal, sampling_rate = librosa.load(audio_path)

# Normalize signal
signal = signal / np.max(np.abs(signal))

# Define autoencoder architecture
latent_dim = 32 # dimension of the latent space

input_shape = (signal.shape[0], 1)

# Encoder
encoder_inputs = layers.Input(shape=input_shape)
x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_inputs)
x = layers.MaxPooling1D(2, padding='same')(x)
x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(2, padding='same')(x)
x = layers.Flatten()(x)
encoder_outputs = layers.Dense(latent_dim, activation='relu')(x)
encoder = models.Model(encoder_inputs, encoder_outputs)

# Decoder
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(np.prod(input_shape[:-1]), activation='relu')(decoder_inputs)
x = layers.Reshape(input_shape)(x)
x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
x = layers.UpSampling1D(2)(x)
decoder_outputs = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)
decoder = models.Model(decoder_inputs, decoder_outputs)

# Autoencoder
autoencoder_inputs = layers.Input(shape=input_shape)
latent_space = encoder(autoencoder_inputs)
autoencoder_outputs = decoder(latent_space)
autoencoder = models.Model(autoencoder_inputs, autoencoder_outputs)

# Compile autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder
history = autoencoder.fit(np.expand_dims(signal, axis=-1),
                          np.expand_dims(signal, axis=-1),
                          epochs=50, batch_size=128,
                          verbose=2)

# Test autoencoder
decoded_signal = autoencoder.predict(np.expand_dims(signal, axis=-1)).squeeze()

# Plot original and reconstructed signals
plt.plot(signal)
plt.plot(decoded_signal)
plt.show()
