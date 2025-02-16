import torch
from torch import nn

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn, dropout_prob, normalize, output_logits=False):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation_fn()

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.normalize = None
        if normalize:
            self.normalize = normalize(output_dim)
        #     self.norm = nn.BatchNorm1d(output_dim)

        self.output_logits = output_logits

    def forward(self, x):
        y = self.linear(x)

        if self.output_logits:
            return y

        if self.normalize:
            y = self.normalize(y)

        y = self.activation(y)

        if self.dropout_prob > 0.0:
            y = self.dropout(y)

        return y

class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=784, activation_fn=torch.nn.ReLU, final_activation_fn=nn.Sigmoid, dropout_prob=0.0, normalize=True, output_logits=False):
        super().__init__()

        self.layers = nn.Sequential(
            LinearBlock(input_dim, 1200, activation_fn, dropout_prob, normalize),
            LinearBlock(1200, 1200, activation_fn, dropout_prob, normalize),
            LinearBlock(1200, output_dim, final_activation_fn, 0.0, None, output_logits=output_logits)
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784, output_dim=1, activation_fn=nn.LeakyReLU, final_activation_fn=nn.Sigmoid, dropout_prob=0.0, normalize=True, output_logits=False):
        super().__init__()
        
        self.layers = nn.Sequential(
            LinearBlock(input_dim, 1200, activation_fn, dropout_prob, normalize),
            LinearBlock(1200, 1200, activation_fn, dropout_prob, normalize),
            LinearBlock(1200, output_dim, final_activation_fn, 0.0, None, output_logits=output_logits)
        )

    def forward(self, x):
        return self.layers(x)
    

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, cond_dim=10, output_dim=784, activation_fn=torch.nn.ReLU, final_activation_fn=nn.Sigmoid, dropout_prob=0.0, normalize=True, output_logits=False):
        super().__init__()

        self.noise_layer = LinearBlock(noise_dim, 200, activation_fn, dropout_prob, normalize)
        self.cond_layer = LinearBlock(cond_dim, 1000, activation_fn, dropout_prob, normalize)

        self.joint_layers = nn.Sequential(
            LinearBlock(1200, 1200, activation_fn, dropout_prob, normalize),
            # LinearBlock(1200, 1200, activation_fn, dropout_prob, normalize),
            LinearBlock(1200, output_dim, final_activation_fn, 0.0, False, output_logits=output_logits)
        )

    def forward(self, x, y):
        h1, h2 = self.noise_layer(x), self.cond_layer(y)
        h = torch.cat((h1, h2), dim=1)
        return self.joint_layers(h)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim=784, cond_dim=10, output_dim=1, activation_fn=nn.LeakyReLU, final_activation_fn=nn.Sigmoid, dropout_prob=0.0, normalize=True, output_logits=False):
        super().__init__()

        self.input_layer = LinearBlock(input_dim, 1200, activation_fn, dropout_prob, normalize)
        self.cond_layer = LinearBlock(cond_dim, 200, activation_fn, dropout_prob, normalize)
        
        self.joint_layers = nn.Sequential(
            LinearBlock(1400, 1200, activation_fn, dropout_prob, normalize),
            # LinearBlock(1200, 1200, activation_fn, dropout_prob, normalize),
            LinearBlock(1200, output_dim, final_activation_fn, 0.0, False, output_logits=output_logits)
        )

    def forward(self, x, y):
        h1, h2 = self.input_layer(x), self.cond_layer(y)
        h = torch.cat((h1, h2), dim=1)
        return self.joint_layers(h)