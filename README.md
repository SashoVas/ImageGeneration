# Image Generation with Deep Learning

This project investigates different ways to generate images with deep learning. At the current stage, the repository focuses on Variational Autoencoders (VAEs), a class of generative models that learn how to represent images in a compact latent space and then use that representation to reconstruct or synthesize new samples.

# 1.  Variational Autoencoders

A Variational Autoencoder is a neural network that combines an encoder and a decoder. The encoder maps an input image into a lower-dimensional latent representation, while the decoder reconstructs the image from that representation. Unlike a standard autoencoder, a VAE does not encode each input to a single point in latent space. Instead, it learns a probability distribution over the latent variables, typically modeled as a Gaussian. This makes the model capable of generating new images by sampling from the latent space rather than simply memorizing training examples.

In this project, the VAE is implemented in [VariationalAutoencoders/model.py](VariationalAutoencoders/model.py) and explored in [VariationalAutoencoders/variational_autoencoder.ipynb](VariationalAutoencoders/variational_autoencoder.ipynb). The goal is to study how well the model can learn meaningful structure from image data and produce visually coherent outputs.

## U-Net Style Architecture

The model uses a convolutional architecture inspired by U-Net principles. The encoder progressively reduces the spatial dimensions of the image while increasing the number of learned feature channels, allowing the network to capture increasingly abstract visual patterns. The decoder then expands these feature maps back to image space, reconstructing the original input through a series of upsampling and convolutional layers.

<img src="readme_images/unet.png" alt="U-Net style architecture" width="360" />

## Generated Samples

<div style="display: flex; flex-wrap: wrap; gap: 8px;">
  <img src="variational_autoencoder_images/vae1.png" alt="Generated sample 1" width="140" />
  <img src="variational_autoencoder_images/vae2.png" alt="Generated sample 2" width="140" />
  <img src="variational_autoencoder_images/vae3.png" alt="Generated sample 3" width="140" />
  <img src="variational_autoencoder_images/vae4.png" alt="Generated sample 4" width="140" />
  <img src="variational_autoencoder_images/vae5.png" alt="Generated sample 5" width="140" />
  <img src="variational_autoencoder_images/vae6.png" alt="Generated sample 6" width="140" />
  <img src="variational_autoencoder_images/vae7.png" alt="Generated sample 7" width="140" />
  <img src="variational_autoencoder_images/vae8.png" alt="Generated sample 7" width="140" />
  <img src="variational_autoencoder_images/vae9.png" alt="Generated sample 7" width="140" />
</div>


