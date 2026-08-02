import torch


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(EncoderBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.pooling = torch.nn.MaxPool2d((2, 2))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        first = self.relu(self.conv1(x))
        second = self.relu(self.conv2(first))
        return self.pooling(second)


class DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(DecoderBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.ReLU()
        self.upsample = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        first = self.relu(self.conv1(x))
        second = self.relu(self.conv2(first))

        return self.upsample(second)


class Encoder(torch.nn.Module):
    def __init__(self, initial_channels, starting_channels=3):
        super(Encoder, self).__init__()
        self.encoder_first = EncoderBlock(starting_channels, initial_channels)
        self.encoder_second = EncoderBlock(
            initial_channels, initial_channels*2)
        self.encoder_third = EncoderBlock(
            initial_channels*2, initial_channels*4)

    def forward(self, x):
        return self.encoder_third(self.encoder_second(self.encoder_first(x)))


class Decoder(torch.nn.Module):
    def __init__(self, initial_channels, ending_channels=3):
        super(Decoder, self).__init__()

        self.decoder_first = DecoderBlock(
            initial_channels*4, initial_channels*2)
        self.decoder_second = DecoderBlock(
            initial_channels*2, initial_channels)
        self.decoder_third = DecoderBlock(initial_channels, initial_channels)
        self.final_conv1 = torch.nn.Conv2d(
            initial_channels, initial_channels, (3, 3), stride=1, padding=1)
        self.final_conv2 = torch.nn.Conv2d(
            initial_channels, ending_channels, (1, 1))
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        first = self.decoder_third(self.decoder_second(self.decoder_first(x)))
        second = self.relu(self.final_conv1(first))
        third = self.final_conv2(second)
        return self.sigmoid(third)


class VAE(torch.nn.Module):
    """
    A simple VAE model that uses convolutional layers for encoding and decoding.
    This implementation of VAE follows the U-Net architecture, with an encoder and decoder that use convolutional layers to process images.
    At the bottleneck, the model learns a latent representation of the input image, which is then used to reconstruct the image in the decoder.
    The bottleneck consists of two linear layers that learn the mean and standard deviation of the latent representation, which are then used to sample from a Gaussian distribution to generate the latent vector.
    """

    def __init__(self, initial_channels, latent_dim, img_width, img_height, ending_channels=3):
        super(VAE, self).__init__()
        self.initial_channels = initial_channels
        self.img_width = img_width
        self.img_height = img_height
        self.ending_channels = ending_channels
        self.encoder = Encoder(
            initial_channels, starting_channels=ending_channels)
        self.decoder = Decoder(
            initial_channels, ending_channels=ending_channels)
        self.mu_layer = torch.nn.Linear(
            initial_channels * 4 * (img_width // 8) * (img_height // 8), latent_dim)
        self.sigma_layer = torch.nn.Linear(
            initial_channels * 4 * (img_width // 8) * (img_height // 8), latent_dim)
        self.to_decoder_layer = torch.nn.Linear(
            latent_dim, initial_channels * 4 * (img_width // 8) * (img_height // 8))

    def decoder_forward(self, z):
        pre_decoder = self.to_decoder_layer(
            z).view(-1, self.initial_channels*4, self.img_width//8, self.img_height//8)
        return self.decoder(pre_decoder)

    def forward(self, x):
        pre_bottleneck = torch.flatten(self.encoder(x), start_dim=1)
        mu = self.mu_layer(pre_bottleneck)
        log_sigma = self.sigma_layer(pre_bottleneck)
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(0.5 * log_sigma)

        z = mu + sigma*epsilon
        result = self.decoder_forward(z)

        return mu, sigma, result


class VAELinear(torch.nn.Module):
    """
    A simple VAE model that uses linear layers instead of convolutional layers.
    Good for small images like MNIST, and for baseline testing of VAE functionality.
    Not suitable for larger images like CIFAR-10.
    """

    def __init__(self, initial_channels, hidden_dim, latent_dim):
        super(VAELinear, self).__init__()
        self.img_to_hidden = torch.nn.Linear(
            initial_channels*initial_channels, hidden_dim)
        self.hidden1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_log_sigma = torch.nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = torch.nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_img = torch.nn.Linear(
            hidden_dim, initial_channels*initial_channels)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        hidden = torch.relu(self.img_to_hidden(x))
        hidden = torch.relu(self.hidden1(hidden))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_log_sigma(hidden)
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(0.5 * log_sigma)

        z = mu + sigma*epsilon

        hidden_decoded = torch.relu(self.latent_to_hidden(z))
        hidden_decoded = torch.relu(self.hidden2(hidden_decoded))
        result = torch.sigmoid(self.hidden_to_img(hidden_decoded))

        return mu, sigma, result.view(-1, 1, 32, 32)
