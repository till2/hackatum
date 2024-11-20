import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, feature_dim: int, feature_map_size: int = None):
        super().__init__()

        # CNN from conv-feature map
        # self.layers = nn.Sequential(
        #     nn.Conv2d(feature_dim, 512, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(128 * (feature_map_size // 8) * (feature_map_size // 8), 1),
        #     nn.Sigmoid()
        # )

        # MLP from penultimate fc output 
        # input shape: [B, feature_dim]
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid() # no sigmoid for wasserstein loss
        )

    def forward(self, x):
        return self.layers(x)
