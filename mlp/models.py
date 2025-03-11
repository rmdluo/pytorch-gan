import torch
from torch import nn

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn, dropout_prob, normalize_fn, activation_args={}, normalize_args={}, output_logits=False):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation_fn(**activation_args)

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.normalize = None
        if normalize_fn:
            self.normalize = normalize_fn(output_dim, **normalize_args)
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

    def weight_init(self, mean, std):
        normal_init(self.linear, mean, std)

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

    def weight_init(self, mean, std):
        self.apply(lambda x : normal_init(x, mean, std))

    def generate_noise(self, batch_size):
        return torch.randn((batch_size, self.noise_dim))

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

    def weight_init(self, mean, std):
        self.apply(lambda x : normal_init(x, mean, std))
    

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, cond_dim=10, output_dim=784, activation_fn=torch.nn.ReLU, activation_args={}, final_activation_fn=nn.Sigmoid, final_activation_args={}, dropout_prob=0.0, normalize=True, normalize_args={}, output_logits=False):
        super().__init__()

        self.noise_dim = noise_dim

        self.noise_layer = LinearBlock(noise_dim, 256, activation_fn, dropout_prob, normalize, activation_args, normalize_args)
        self.cond_layer = LinearBlock(cond_dim, 256, activation_fn, dropout_prob, normalize, activation_args, normalize_args)

        self.joint_layers = nn.Sequential(
            LinearBlock(512, 512, activation_fn, dropout_prob, normalize, activation_args, normalize_args),
            LinearBlock(512, 1024, activation_fn, dropout_prob, normalize, activation_args, normalize_args),
            LinearBlock(1024, output_dim, final_activation_fn, 0.0, False, final_activation_args, output_logits=output_logits)
        )

    def forward(self, x, y):
        h1, h2 = self.noise_layer(x), self.cond_layer(y)
        h = torch.cat((h1, h2), dim=1)
        return self.joint_layers(h)

    def weight_init(self, mean, std):
        self.apply(lambda x : normal_init(x, mean, std))

    def generate_noise(self, batch_size):
        return torch.randn((batch_size, self.noise_dim))

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim=784, cond_dim=10, output_dim=1, activation_fn=nn.LeakyReLU, activation_args={}, final_activation_fn=nn.Sigmoid, final_activation_args={}, dropout_prob=0.0, normalize=True, normalize_args={}, output_logits=False):
        super().__init__()

        self.input_layer = LinearBlock(input_dim, 1024, activation_fn, dropout_prob, False, activation_args)
        self.cond_layer = LinearBlock(cond_dim, 1024, activation_fn, dropout_prob, False, activation_args)
        
        self.joint_layers = nn.Sequential(
            LinearBlock(2048, 512, activation_fn, dropout_prob, normalize, activation_args, normalize_args),
            LinearBlock(512, 256, activation_fn, dropout_prob, normalize, activation_args, normalize_args),
            LinearBlock(256, output_dim, final_activation_fn, 0.0, False, final_activation_args, output_logits=output_logits)
        )

    def forward(self, x, y):
        h1, h2 = self.input_layer(x), self.cond_layer(y)
        h = torch.cat((h1, h2), dim=1)
        return self.joint_layers(h)

    def weight_init(self, mean, std):
        self.apply(lambda x : normal_init(x, mean, std))
