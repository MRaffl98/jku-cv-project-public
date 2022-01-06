import torch.nn as nn

class PCA(nn.Module):
    def __init__(self, n_components=10, **kwargs):
        super().__init__()
        self.encoder_layer = nn.Linear(in_features=kwargs["input_shape"], out_features=n_components)
        self.decoder_layer = nn.Linear(in_features=n_components, out_features=kwargs["input_shape"])

    def forward(self, features):
        latent = self.encoder_layer(features)
        output = self.decoder_layer(latent)
        return output
