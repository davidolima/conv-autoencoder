import torch
import torch.nn as nn

from math import ceil

class Encoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            input_channels:int,
            bottleneck_dim:int,
            hidden_channels:int = 8
    ) -> None:
        super().__init__()

        _n_modules = 5

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # B x C x M x M -> B x C*2^(_n_modules) x  M/2^(_n_modules) x M/2^(_n_modules)
        # B x 3 x 224 x 224 -> B x 8192 x 4 x 4
        for i in range(_n_modules):
            self.encoder.add_module(
                name=f"ConvBlock-{ceil(input_size/(2**i))}->{ceil(input_size/(2**(i+1)))}-" + str(i),
                module=self._conv_block((2**i)*hidden_channels, (2**(i+1))*hidden_channels)
            )

        # This is what we get after the convolutions
        hidden_channels *= (2**_n_modules)
        hidden_size = ceil(input_size/(2**_n_modules))

        # output
        self.encoder.add_module(
            name="output",
            module=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*hidden_size*hidden_channels, bottleneck_dim),
        ))


    def _conv_block(self, input_channels:int, output_channels:int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d((2,2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        #print("Encoder output shape:", x.shape)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            input_channels:int,
            bottleneck_dim:int,
            hidden_channels:int = 8,
    ) -> None:
        super().__init__()

        _n_modules = 5
        input_dim = ceil(input_size/(2**_n_modules))
        input_dim *= input_dim
        input_dim *= hidden_channels*(2**_n_modules)

        self.linear = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.ReLU(),
        )


        self.decoder = nn.Sequential()

        # B x C x M x M -> B x C/2^(_n_modules) x  M*2^(_n_modules) x M*2^(_n_modules)
        # B x 8192 x 4 x 4 -> B x 8 x 224 x 224
        for i in range(_n_modules, 0, -1):
            self.decoder.add_module(
                name=f"DeconvBlock-{ceil(input_size/(2**i))}->{ceil(input_size/(2**(i-1)))}-" + str(_n_modules-i),
                module=self._deconv_block((2**i)*hidden_channels, (2**(i-1))*hidden_channels)
            )

        self.decoder.add_module(
            name="output",
            module=nn.Sequential(
                nn.Conv2d(hidden_channels, input_channels, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        )

    def _deconv_block(self, input_channels:int, output_channels:int)->nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.view(x.shape[0], -1, 7, 7)
        x = self.decoder(x)
        #print("Decoder ouput shape:", x.shape)
        return x

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        bottleneck_dim:int,
        input_channels:int = 3,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_size=input_size,
            input_channels=input_channels,
            bottleneck_dim=bottleneck_dim,
        )

        self.decoder = Decoder(
            input_size=input_size,
            input_channels=input_channels,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    img_shape = (1,3,224,224)
    ae = Autoencoder(
        input_size=img_shape[2],
        bottleneck_dim=128,
        input_channels=img_shape[1]
    )
    print(ae)
    noise = torch.rand(img_shape)
    out = ae(noise)
    print(out.shape, "OK!" if noise.shape == out.shape else f"\nWARNING: Output shape differs from input shape. Should have been: {noise.shape}")
    print(out)
