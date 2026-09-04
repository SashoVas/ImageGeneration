
import torch
import torch.nn.functional as F
import torch


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, num_groups=16, use_attention=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # self.pool = torch.nn.MaxPool2d(2)
        self.pool = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(num_groups, in_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups, out_channels)
        self.embedding = torch.nn.Linear(time_embed_dim, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

        self.shortcut = (
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )
        self.attn = SelfAttention2d(out_channels) if use_attention else None

    def forward(self, x, time):
        residual = self.shortcut(x)

        embed = self.embedding(time).unsqueeze(-1).unsqueeze(-1)

        x_first = self.conv1(self.relu(self.norm1(x)))
        x_first = x_first + embed
        if self.attn is not None:
            x_first = self.attn(x_first)
        x_second = self.conv2(self.dropout(self.relu(self.norm2(x_first))))
        x = x_second + residual  # residual connection

        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, num_groups=16, use_attention=False):
        super().__init__()
        # self.up = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.up = torch.nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(in_channels*2, out_channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = torch.nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(num_groups, in_channels*2)
        self.norm2 = torch.nn.GroupNorm(num_groups, out_channels)
        self.embedding = torch.nn.Linear(time_embed_dim, out_channels)
        self.shortcut = (
            torch.nn.Conv2d(in_channels*2, out_channels, kernel_size=1)
            if in_channels*2 != out_channels
            else torch.nn.Identity()
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.attn = SelfAttention2d(out_channels) if use_attention else None

    def forward(self, x, skip, time):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        residual = self.shortcut(x)

        embed = self.embedding(time).unsqueeze(-1).unsqueeze(-1)

        x_first = self.conv1(self.relu(self.norm1(x)))
        x_first = x_first + embed
        if self.attn is not None:
            x_first = self.attn(x_first)
        x_second = self.conv2(self.dropout(self.relu(self.norm2(x_first))))
        x = x_second + residual  # residual connection

        return x


class Encoder(torch.nn.Module):
    def __init__(self, initial_channels, time_embed_dim, channel_mults, starting_channels=1, num_blocks=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        current_channels = starting_channels
        for i in range(num_blocks):
            out_channels = initial_channels * channel_mults[i]
            self.blocks.append(EncoderBlock(
                current_channels, out_channels, time_embed_dim, use_attention=(i == 3 or i == 4)))
            current_channels = out_channels

    def forward(self, x, time):
        skips = []
        for block in self.blocks:
            x, skip = block(x, time)
            skips.append(skip)
        return x, skips


class Decoder(torch.nn.Module):
    def __init__(self, initial_channels, time_embed_dim, channel_mults, ending_channels=1, num_blocks=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        current_in = initial_channels * channel_mults[-1]
        for i in range(num_blocks-2, -1, -1):
            out_channels = initial_channels * channel_mults[i]

            self.blocks.append(DecoderBlock(
                current_in, out_channels, time_embed_dim, use_attention=(i == 3 or i == 4)))
            current_in = out_channels
        self.blocks.append(DecoderBlock(
            initial_channels, initial_channels, time_embed_dim, use_attention=False))
        self.final_conv = torch.nn.Conv2d(initial_channels, ending_channels, 1)

    def forward(self, x, time, skips):
        skips = skips[::-1]
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i], time)
        return self.final_conv(x)


class SelfAttention2d(torch.nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = torch.nn.GroupNorm(8, channels)
        self.qkv = torch.nn.Conv2d(
            channels, channels * 3, kernel_size=1, bias=False)
        self.proj = torch.nn.Conv2d(
            channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # q = q.view(b, self.num_heads, self.head_dim, h * w)
        # k = k.view(b, self.num_heads, self.head_dim, h * w)
        # v = v.view(b, self.num_heads, self.head_dim, h * w)

        # q = q.permute(0, 1, 3, 2)
        # attn = torch.matmul(q, k) / (self.head_dim ** 0.5)
        # attn = torch.softmax(attn, dim=-1)
        # out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        # out = torch.nn.functional.scaled_dot_product_attention(
        #    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        # out = out.permute(0, 1, 3, 2).contiguous()
        # out = out.view(b, c, h, w)
        q = q.view(b, self.num_heads, self.head_dim, h *
                   w).transpose(-1, -2)  # (b, heads, hw, head_dim)
        k = k.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).contiguous().view(b, c, h, w)
        out = self.proj(out)
        return out + residual


class UNet(torch.nn.Module):
    def __init__(self, initial_channels, time_embed_dim, channel_mults, ending_channels=3, num_blocks=3):
        super(UNet, self).__init__()
        self.conv_in = torch.nn.Conv2d(
            ending_channels, initial_channels, kernel_size=3, padding=1)
        self.encoder = Encoder(initial_channels, time_embed_dim, channel_mults,
                               starting_channels=initial_channels, num_blocks=num_blocks)
        self.decoder = Decoder(initial_channels, time_embed_dim, channel_mults,
                               ending_channels=ending_channels, num_blocks=num_blocks)
        # self.bottleneck = torch.nn.Conv2d(initial_channels*4, initial_channels*4, kernel_size=(3,3), stride=1, padding=1)
        bottleneck_channels = initial_channels * channel_mults[-1]
        self.bottleneck_first = torch.nn.Sequential(
            torch.nn.GroupNorm(16, bottleneck_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(bottleneck_channels, bottleneck_channels,
                            kernel_size=3, stride=1, padding=1),
        )
        # self.attn = SelfAttention2d(bottleneck_channels, num_heads=16)
        self.bottleneck_second = torch.nn.Sequential(
            torch.nn.GroupNorm(16, bottleneck_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(bottleneck_channels, bottleneck_channels,
                            kernel_size=3, stride=1, padding=1),

        )

    def forward(self, x, time):
        x = self.conv_in(x)

        encoder_output, skips = self.encoder(x, time)
        z_first = self.bottleneck_first(encoder_output)
        z_first = z_first + encoder_output  # residual connection
        # z_second = self.attn(z_first)
        z_second = self.bottleneck_second(z_first)
        z_second = z_second + z_first  # residual connection
        decoder_output = self.decoder(z_second, time, skips=skips)
        return decoder_output


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, t):
        pos = torch.floor(t * self.max_len).long().clamp(0, self.max_len - 1)
        return self.pe[pos]


class FlowMatchingModel(torch.nn.Module):
    def __init__(self, initial_channels, channel_mults, time_embed_dim=32, ending_channels=3, num_blocks=3, t_bins=300):
        super(FlowMatchingModel, self).__init__()
        self.ending_channels = ending_channels
        self.t_bins = t_bins
        self.positional_encoding = torch.nn.Sequential(
            PositionalEncoding(time_embed_dim, max_len=t_bins),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, time_embed_dim),
            torch.nn.SiLU()
        )
        self.unet = UNet(initial_channels, time_embed_dim, channel_mults,
                         ending_channels=ending_channels, num_blocks=num_blocks)

    def forward(self, x0, t):
        time = self.positional_encoding(t)

        velocity = self.unet(x0, time)
        return velocity

    def sample(self, img_size, num_samples, device, bins=None):
        if bins is None:
            bins = self.t_bins
        x = torch.randn(num_samples, self.ending_channels,
                        img_size, img_size).to(device)
        for t in torch.linspace(0.0, 1.0, bins):
            t_tensor = torch.full((num_samples,), t.item(),
                                  dtype=torch.float32, device=device)
            velocity = self.forward(x, t_tensor)
            x = x + (1.0 / bins) * velocity
        return x
