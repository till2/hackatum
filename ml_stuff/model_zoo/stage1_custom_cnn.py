import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.identity_conv = None

    def forward(self, x):
        identity = x

        if self.identity_conv is not None:
            identity = self.identity_conv(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class DownsampleHeight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 3),
            stride=(stride, 1),
            padding=(padding, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SqueezeHeight(nn.Module):
    def __init__(self, in_channels=3, init_features=64, kernel_size=7, stride=2, padding=2):
        super().__init__()
        num_layers = 7

        self.layers = nn.ModuleList()
        features = init_features

        for i in range(num_layers):
            self.layers.append(self._make_resnet_blocks(in_channels, features))
            if i < num_layers - 1:
                self.layers.append(DownsampleHeight(features, features, kernel_size, stride, padding))
            else:
                # final layer
                self.layers.append(DownsampleHeight(features, features, kernel_size + 2, stride, padding))

            in_channels = features

            if i > 3:
                features *= 2

    def _make_resnet_blocks(self, in_channels, out_channels, blocks=1):
        layers = []
        for _ in range(blocks):
            layers.append(ResNetBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super().__init__()

        # Encoding path
        self.enc1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1)
        self.enc4 = nn.Conv1d(in_channels * 4, in_channels * 8, kernel_size=3, padding=1)
        self.enc5 = nn.Conv1d(in_channels * 8, in_channels * 16, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Decoding path
        self.upconv1 = nn.ConvTranspose1d(in_channels * 16, in_channels * 8, kernel_size=2, stride=2)
        self.dec1 = nn.Conv1d(in_channels * 16, in_channels * 8, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose1d(in_channels * 8, in_channels * 4, kernel_size=2, stride=2)
        self.dec2 = nn.Conv1d(in_channels * 8, in_channels * 4, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose1d(in_channels * 4, in_channels * 2, kernel_size=2, stride=2)
        self.dec3 = nn.Conv1d(in_channels * 4, in_channels * 2, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose1d(in_channels * 2, in_channels, kernel_size=2, stride=2)
        self.dec4 = nn.Conv1d(in_channels * 2, in_channels, kernel_size=3, padding=1)

        self.final_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.maxpool(e1)))
        e3 = F.relu(self.enc3(self.maxpool(e2)))
        e4 = F.relu(self.enc4(self.maxpool(e3)))
        e5 = F.relu(self.enc5(self.maxpool(e4)))

        # Decoder
        d1 = F.relu(self.dec1(torch.cat((self.upconv1(e5), e4), dim=1)))
        d2 = F.relu(self.dec2(torch.cat((self.upconv2(d1), e3), dim=1)))
        d3 = F.relu(self.dec3(torch.cat((self.upconv3(d2), e2), dim=1)))
        d4 = F.relu(self.dec4(torch.cat((self.upconv4(d3), e1), dim=1)))
        out = self.final_conv(d4)

        return out


class Stage1EdgeDetector(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=64):
        super().__init__()
        self.squeeze_height = SqueezeHeight(in_channels=in_channels, init_features=init_features)
        self.unet1d = UNet1D(in_channels=init_features * 4, out_channels=out_channels)

    def forward(self, x):
        x = self.squeeze_height(x)
        x = rearrange(x, "b c 1 w -> b c w")
        x = self.unet1d(x)
        return x  # b c=2 w


class HackyModel(nn.Module):
    """Only reduces the height (the output has local width context)."""

    def __init__(self, in_channels=3, out_channels=2, init_features=64):
        super().__init__()
        self.squeeze_height = SqueezeHeight(in_channels=in_channels, init_features=init_features)
        self.final_conv = nn.Conv1d(
            in_channels=init_features * 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.squeeze_height(x)
        x = rearrange(x, "b c 1 w -> b c w")
        x = self.final_conv(x)
        return x


class SharedLinear(nn.Module):
    def __init__(
        self,
        in_channels=128,
        in_height=14,
        in_width=14,
        out_channels=64,
        out_width=448,
        out_height=2,
    ):
        super().__init__()

        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width

        # input shape: (B,C,H,W), e.g. (2,128,14,14)
        # output shape: (B,C',H'=2,W'=448), e.g. (2,64,2,448)
        # this function applies the same linear layer across the input width dimension (e.g. apply it 14 times).
        # each layer transforms: (B,C*H) -> (B,C'*H'*(W'/W))

        self.linear_input_size = in_channels * in_height
        self.linear_output_size = out_channels * out_height * (out_width // in_width)

        # Define the shared linear layer
        self.shared_linear = nn.Linear(self.linear_input_size, self.linear_output_size)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.in_height and W == self.in_width, "Input tensor does not match the expected dimensions."

        transformed_slices = []
        for i in range(W):
            x_slice = x[:, :, :, i]  # x_slice shape: (B, C, H)
            x_slice_flattened = x_slice.view(B, -1)  # (B, C*H)

            transformed = self.shared_linear(x_slice_flattened)  # (B, C'*H'*(W'/W))
            transformed = rearrange(
                transformed,
                "B (C H W) -> B C H W",
                C=self.out_channels,
                H=self.out_height,
                W=(self.out_width // self.in_width),
            )

            transformed_slices.append(transformed)

        output = torch.cat(transformed_slices, dim=-1)  # Concatenate along new width dimension
        return output


class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # local model
        self.stage1_local_kv_model = SqueezeHeight(in_channels=8, init_features=64)

        # global model
        self.stage1_global_q_model = timm.create_model("swsl_resnext101_32x8d", pretrained=True)
        self.stage1_global_q_model.conv1 = nn.Conv2d(
            in_channels=8,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        # replace head of resnet
        del self.stage1_global_q_model.global_pool
        del self.stage1_global_q_model.fc
        self.stage1_global_q_model.conv1x1 = nn.Conv2d(
            in_channels=2048,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.stage1_global_q_model.shared_linear = SharedLinear(
            in_channels=128,
            in_height=14,
            in_width=14,
            out_channels=64,
            out_width=448,
            out_height=2,
        )

        def forward_head(this, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
            x = this.conv1x1(x)  # change channels from 2048 -> 128
            x = this.shared_linear(x)

            if this.drop_rate:
                x = F.dropout(x, p=float(this.drop_rate), training=this.training)
            return x

        def forward(this, x: torch.Tensor) -> torch.Tensor:
            x = this.forward_features(x)
            x = this.forward_head(x)
            return x

        self.stage1_global_q_model.forward_head = forward_head.__get__(self.stage1_global_q_model)
        self.stage1_global_q_model.forward = forward.__get__(self.stage1_global_q_model)

        # combine the two model outputs
        self.multihead_attn = nn.MultiheadAttention(embed_dim=448, num_heads=8, batch_first=True)
        self.conv_1x1 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, bias=False)

    def forward(self, x):
        q = self.stage1_global_q_model(x)
        start_q, end_q = torch.split(q, (1, 1), dim=2)  # (B,C,1,W)
        start_q, end_q = start_q.squeeze(2), end_q.squeeze(2)  # (B,C,W) for start_q and end_q

        kv = self.stage1_local_kv_model(x)
        kv = rearrange(kv, "b c 1 w -> b c w")
        start_k, start_v, end_k, end_v = torch.split(kv, (64, 64, 64, 64), dim=1)  # (B,C,W) for start_k, start_v, end_k, end_v

        # Apply Multihead Attention
        start_logits, _ = self.multihead_attn(start_q, start_k, start_v)  # (B, W=448, 448)
        end_logits, _ = self.multihead_attn(end_q, end_k, end_v)  # (B, W=448, 448)

        start_logits = self.conv_1x1(start_logits).squeeze(1)  # (B, W)
        end_logits = self.conv_1x1(end_logits).squeeze(1)  # (B, W)

        return start_logits, end_logits
