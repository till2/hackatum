import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, kernel_size=3, stride=1, padding=1, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(
                ch_out,
                ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, t=2, num_rec_blocks=2):
        super().__init__()

        rec_blocks = [Recurrent_block(ch_out, kernel_size, stride, padding, t=t) for _ in range(num_rec_blocks)]
        self.RCNN = nn.Sequential(*rec_blocks)

        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv_1d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class RRCNN_block_1d(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(*[Recurrent_block_1d(ch_out, t=t) for _ in range(2)])
        self.Conv_1x1 = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Recurrent_block_1d(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class R2Autoencoder(nn.Module):
    def __init__(self, img_ch_in=3, img_ch_out=3, init_features=32, latent_dim=1024, t=2, supervised_head=None):
        super().__init__()
        self.ftrs = init_features

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.supervised_head = supervised_head

        # Encoder
        self.RRCNN1 = RRCNN_block(ch_in=img_ch_in, ch_out=self.ftrs, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=self.ftrs, ch_out=self.ftrs * 2, t=t)
        self.RRCNN3 = RRCNN_block(ch_in=self.ftrs * 2, ch_out=self.ftrs * 4, t=t)
        self.RRCNN4 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)  # cap at ftrs*4
        self.RRCNN5 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)
        self.RRCNN6 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)
        # Latent space
        self.fc1 = nn.Linear(self.ftrs * 4 * 14 * 14, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.ftrs * 4 * 14 * 14) # TODO: make this smaller and have more conv layers in the decoderinstead (the layers goes from 1024 -> 25088)

        # Decoder
        self.Up5 = up_conv(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)
        self.Up4 = up_conv(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4)
        self.Up_RRCNN4 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)
        self.Up3 = up_conv(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4)
        self.Up_RRCNN3 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)
        self.Up2 = up_conv(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4)
        self.Up_RRCNN2 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4, t=t)
        self.Up1 = up_conv(ch_in=self.ftrs * 4, ch_out=self.ftrs * 4)
        self.Up_RRCNN1 = RRCNN_block(ch_in=self.ftrs * 4, ch_out=img_ch_out, t=t)

        self._init_weights()

    def forward(self, x, verbose=False, predict_labels=False):
        # Encoding path
        x = self.RRCNN1(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN2(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN3(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN4(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN5(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN6(x)
        if verbose:
            print(x.shape)

        # Flatten and pass through latent space
        x = x.view(x.size(0), -1)
        latent = self.fc1(x)

        if verbose:
            print(latent.shape)
        if predict_labels and self.supervised_head is not None:
            return self.supervised_head(latent)

        x = self.fc2(latent)
        x = x.view(x.size(0), self.ftrs * 4, 14, 14)
        if verbose:
            print(x.shape)

        # Decoding path (reverse of encoding)
        x = self.Up5(x)
        x = self.Up_RRCNN5(x)
        if verbose:
            print(x.shape)

        x = self.Up4(x)
        x = self.Up_RRCNN4(x)
        if verbose:
            print(x.shape)

        x = self.Up3(x)
        x = self.Up_RRCNN3(x)
        if verbose:
            print(x.shape)

        x = self.Up2(x)
        x = self.Up_RRCNN2(x)
        if verbose:
            print(x.shape)

        x = self.Up1(x)
        x = self.Up_RRCNN1(x)
        if verbose:
            print(x.shape)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x, verbose=False):
        # Encoding path
        x = self.RRCNN1(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN2(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN3(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN4(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN5(x)
        if verbose:
            print(x.shape)

        x = self.Maxpool(x)
        x = self.RRCNN6(x)
        if verbose:
            print(x.shape)

        # Flatten and pass through latent space
        x = x.flatten(start_dim=1)
        latent = self.fc1(x)
        if verbose:
            print(latent.shape)

        return latent


class AutoencoderHead(nn.Module):
    def __init__(self, latent_dim, img_size, init_features=64, t=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.ftrs = init_features

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, self.ftrs * (img_size // 32)),
        )

        self.RRCNN1 = RRCNN_block_1d(self.ftrs, self.ftrs, t=t)
        self.RRCNN2 = RRCNN_block_1d(self.ftrs, self.ftrs, t=t)
        self.RRCNN3 = RRCNN_block_1d(self.ftrs, self.ftrs, t=t)
        self.RRCNN4 = RRCNN_block_1d(self.ftrs, self.ftrs // 2, t=t)
        self.RRCNN5 = RRCNN_block_1d(self.ftrs // 2, self.ftrs // 4, t=t)

        self.conv_1x1 = nn.Conv1d(self.ftrs // 4, 2, kernel_size=1, padding=(img_size - (img_size // 32) * 32) // 2)

        self.up_conv_1d1 = up_conv_1d(self.ftrs, self.ftrs)
        self.up_conv_1d2 = up_conv_1d(self.ftrs, self.ftrs)
        self.up_conv_1d3 = up_conv_1d(self.ftrs, self.ftrs)
        self.up_conv_1d4 = up_conv_1d(self.ftrs, self.ftrs)
        self.up_conv_1d5 = up_conv_1d(self.ftrs // 2, self.ftrs // 2)

    def forward(self, x, verbose=False):
        x = x.flatten(start_dim=1)
        if verbose:
            print(x.shape)
        x = self.mlp(x)
        if verbose:
            print(x.shape)
        x = x.view(-1, self.ftrs, self.img_size // 32)
        if verbose:
            print(x.shape)

        x = self.up_conv_1d1(x)
        x = self.RRCNN1(x)
        if verbose:
            print(x.shape)
        x = self.up_conv_1d2(x)
        x = self.RRCNN2(x)
        if verbose:
            print(x.shape)
        x = self.up_conv_1d3(x)
        x = self.RRCNN3(x)
        if verbose:
            print(x.shape)
        x = self.up_conv_1d4(x)
        x = self.RRCNN4(x)
        if verbose:
            print(x.shape)
        x = self.up_conv_1d5(x)
        x = self.RRCNN5(x)
        if verbose:
            print(x.shape)

        x = self.conv_1x1(x)
        if verbose:
            print(x.shape)

        start_logits, end_logits = x[:, 0], x[:, 1]
        return start_logits, end_logits


class AutoencoderSupervisedHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        raise DeprecationWarning("This class is deprecated. Use AutoencoderHead instead.")
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_hid = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.activation(self.fc_in(x))
        for layer in self.fc_hid:
            x = self.activation(layer(x))

        return self.fc_out(x)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder = R2Autoencoder().to(device)

    with torch.no_grad():
        x = torch.randn(1, 3, 448, 448).to(device)
        print("Autoencoder:")
        x_hat = autoencoder(x, verbose=True)

        print("\nAutoencoder Only Encode:")
        latent = autoencoder.encode(x, verbose=True)

        autoencoder_head = AutoencoderHead(latent_dim=1024, img_size=448, init_features=64, t=2).to(device)
        print("\nAutoencoder Head:")
        start_logits, end_logits = autoencoder_head(latent, verbose=True)
        print(start_logits.shape, end_logits.shape)

if __name__ == "__main__":
    main()
