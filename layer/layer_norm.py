import torch
import torch.nn as nn


class Layer_Norm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.eps = 1e-9
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))


    def forward(self, x):
        
        mean_x = x.mean(dim=-1, keepdim=True)
        var_x = x.var(dim=-1, keepdim=True, unbiased=False)

        std_x = torch.sqrt(var_x)

        x_norm = (x - mean_x) / (std_x + self.eps)

        x_norm = self.scale * x_norm + self.shift

        return x_norm



class Batch_Norm(nn.Module):
    def __init__(self, num_features, eps=1e-6, momentum=0.1, training=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = training

        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("moving_mean", torch.zeros(num_features))
        self.register_buffer("moving_var", torch.ones(num_features))

    def forward(self, x):
        
        if x.dim() == 2:
            # [N, C] format (Fully Connected)
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            dims = (0,)
            shape = (1, -1)

        elif x.dim() == 4:
            # [N, C, H, W] format (CNN)
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1).
            dims = (0, 2, 3)
            shape = (1, -1, 1, 1)

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        if self.training:
                
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)

            moving_mean = (1.0 - self.momentum) * self.moving_mean + self.momentum * mean.view(-1)
            moving_var = (1.0 - self.momentum) * self.moving_var + self.momentum * var.view(-1)

        else:
            mean = moving_mean.view(shape)
            var = moving_var.view(shape)
            
        
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = self.scale * x_hat + self.shift  # Scale and shift

        return x_norm
