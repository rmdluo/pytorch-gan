# torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
from utils import mnist_to_one_hot

# tqdm import
from tqdm import tqdm

# matplotlib import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MLP GAN imports
from mlp.models import ConditionalGenerator, ConditionalDiscriminator

# misc imports
import math
import os, shutil
import json
import random


# Set hyperparameters
hyperparameters = {
    # train settings
    'batch_size': 128,
    'num_workers': 4,
    'num_discriminator_steps': 1,
    'num_generator_steps': 1,
    'mock_real': True,
    'epochs': 200,

    # optimizer settings
    'discriminator_optimizer': torch.optim.Adam,
    'discriminator_optimizer_settings': {
        'lr': 2e-4,
        'betas': (0.5, 0.999),
    },
    'generator_optimizer': torch.optim.Adam,
    'generator_optimizer_settings': {
        'lr': 2e-4,
        'betas': (0.5, 0.999),
    },

    # lr scheduler settings
    'lr_scheduler': None,
    'lr_scheduler_settings': {},
    'min_lr': 0,

    # architecture
    'noise_dim': 100,
    'cond_dim': 10,
    'generator_activation': torch.nn.ReLU,
    'generator_activation_args': {},
    'generator_final_activation': torch.nn.Tanh,
    'generator_final_activation_args': {},
    'generator_dropout_prob': 0.0,
    'generator_normalize': torch.nn.BatchNorm1d,
    'generator_normalize_args': {},
    'discriminator_activation': torch.nn.LeakyReLU,
    'discriminator_activation_args': {'negative_slope': 0.2},
    'discriminator_final_activation': torch.nn.Sigmoid,
    'discriminator_final_activation_args': {},
    'discriminator_dropout_prob': 0.0,
    'discriminator_normalize': torch.nn.BatchNorm1d,
    'discriminator_normalize_args': {},

    # example image output settings
    'num_images': 4,
    'save_directory': 'conditional_mock_real'
}
index = 0
original = hyperparameters['save_directory']
while os.path.exists(hyperparameters['save_directory']):
    hyperparameters['save_directory'] = original + "_" + str(index)
    index += 1
os.mkdir(hyperparameters['save_directory'])
serializable_config = {key: str(value) if isinstance(value, type) else value for key, value in hyperparameters.items()}
with open(os.path.join(hyperparameters['save_directory'], "config.json"), "w") as file:
    json.dump(serializable_config, file, indent=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5)), transforms.Lambda(lambda x: x.flatten())])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_val = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
image_shape = mnist_train[0][0].shape[0]

train_dl = DataLoader(
    mnist_train,
    batch_size=hyperparameters['batch_size'],
    num_workers=hyperparameters['num_workers'],
    pin_memory=True,
    shuffle=True,
)

val_dl = DataLoader(
    mnist_val,
    batch_size=hyperparameters['batch_size'],
    num_workers=hyperparameters['num_workers'],
    pin_memory=True,
    shuffle=False,
)


# instantiate models
generator = ConditionalGenerator(
    hyperparameters['noise_dim'],
    hyperparameters['cond_dim'],
    image_shape,
    hyperparameters['generator_activation'],
    hyperparameters['generator_activation_args'],
    hyperparameters['generator_final_activation'],
    hyperparameters['generator_final_activation_args'],
    hyperparameters['generator_dropout_prob'],
    hyperparameters['generator_normalize'],
    hyperparameters['generator_normalize_args'],
)
generator.weight_init(0, 0.02)
generator.to(device)
print("###########################  GENERATOR  ##########################")
summary(generator, input_size=[tuple([hyperparameters['noise_dim']]), tuple([hyperparameters['cond_dim']])])
with open(os.path.join(hyperparameters['save_directory'], "arch_generator.txt"), mode='w') as f:
    f.write(str(generator))

discriminator = ConditionalDiscriminator(
    image_shape,
    hyperparameters['cond_dim'],
    1,
    hyperparameters['discriminator_activation'],
    hyperparameters['discriminator_activation_args'],
    hyperparameters['discriminator_final_activation'],
    hyperparameters['discriminator_final_activation_args'],
    hyperparameters['discriminator_dropout_prob'],
    hyperparameters['discriminator_normalize'],
    hyperparameters['discriminator_normalize_args'],
)
discriminator.weight_init(0, 0.02)
discriminator.to(device)
print("#######################  DISCRIMINATOR  ########################")
summary(discriminator, input_size=[tuple([image_shape]), tuple([hyperparameters['cond_dim']])])
with open(os.path.join(hyperparameters['save_directory'], "arch_discriminator.txt"), mode='w') as f:
    f.write(str(discriminator))

# set up loss_fn
loss_fn = torch.nn.BCELoss()

# set up optimizers
generator_optimizer = hyperparameters['generator_optimizer'](generator.parameters(), **hyperparameters['generator_optimizer_settings'])
discriminator_optimizer = hyperparameters['discriminator_optimizer'](discriminator.parameters(), **hyperparameters['discriminator_optimizer_settings'])

# set up learning rate scheduler
if hyperparameters['lr_scheduler']:
    generator_lr_scheduler = hyperparameters['lr_scheduler'](generator_optimizer, **hyperparameters['lr_scheduler_settings'])
    discriminator_lr_scheduler = hyperparameters['lr_scheduler'](discriminator_optimizer, **hyperparameters['lr_scheduler_settings'])

def print_train_statistics(discriminator_combined_loss, discriminator_real_loss, discriminator_fake_loss, generator_loss):
    print('Train statistics:')
    print(f'\t{"Discriminator combined loss:":<30}{discriminator_combined_loss:.04f}')
    print(f'\t{"Discriminator real loss:":<30}{discriminator_real_loss:.04f}')
    print(f'\t{"Discriminator fake loss:":<30}{discriminator_fake_loss:.04f}')
    print(f'\t{"Generator loss:":<30}{generator_loss:.04f}')

def print_validation_statistics(discriminator_combined_loss, discriminator_real_loss, discriminator_fake_loss, generator_loss, real_correctness_rate, generator_fooling_rate):
    print('Validation statistics:')
    print(f'\t{"Discriminator combined loss:":<30}{discriminator_combined_loss:.04f}')
    print(f'\t{"Discriminator real loss:":<30}{discriminator_real_loss:.04f}')
    print(f'\t{"Discriminator fake loss:":<30}{discriminator_fake_loss:.04f}')
    print(f'\t{"Generator loss:":<30}{generator_loss:.04f}')
    print(f'\t{"Real correctness rate:":<30}{real_correctness_rate:.04f}%')
    print(f'\t{"Generator fooling rate:":<30}{generator_fooling_rate:.04f}%')

def visualize_mnist(generator):
    size = 10

    generator.eval()
    imgs = generator(
        generator.generate_noise(size).to(device),
        mnist_to_one_hot([i for i in range(size)]).to(device)
    )
    img_size = int(math.sqrt(imgs[0].shape[-1]))

    fig, axes = plt.subplots(1, size)

    for idx in range(10):
        img = torch.clamp(imgs[idx], 0, 255).cpu().detach().numpy().reshape(img_size, img_size)  # Assuming square
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(idx, fontsize=10)
        axes[idx].axis('off')
        
    plt.savefig(os.path.join(hyperparameters['save_directory'], f"generated_epoch_{epoch}.png"))
    plt.close()

def train_discriminator_one_step(generator, discriminator, x, y, loss_fn, discriminator_optimizer, mock_real=True):
    batch_size = x.shape[0]

    # zero out gradients
    discriminator_optimizer.zero_grad()

    # generate examples
    noise = generator.generate_noise(batch_size).to(device)
    one_hot = mnist_to_one_hot(y).to(device)
    if mock_real:
        one_hot_fake = one_hot
    else:
        one_hot_fake = mnist_to_one_hot(torch.randint(0, 10, (batch_size,))).to(device)
    x_generated = generator(noise, one_hot_fake)

    # generate loss from real examples
    y_real = discriminator(x, one_hot)
    real_loss = loss_fn(y_real, torch.ones(batch_size, 1).to(device))

    # generate loss from fake examples
    y_fake = discriminator(x_generated, one_hot_fake)
    fake_loss = loss_fn(y_fake, torch.zeros(batch_size, 1).to(device))

    # combine loss
    loss = real_loss + fake_loss

    # backprop
    loss.backward()
    discriminator_optimizer.step()

    return loss.item(), real_loss.item(), fake_loss.item()

def train_generator_one_step(generator, discriminator, x, y, loss_fn, generator_optimizer, mock_real=True):
    batch_size = x.shape[0]

    # zero out gradients
    generator_optimizer.zero_grad()

    # generate examples
    noise = generator.generate_noise(batch_size).to(device)
    one_hot = mnist_to_one_hot(y).to(device)
    if mock_real:
        one_hot_fake = one_hot
    else:
        one_hot_fake = mnist_to_one_hot(torch.randint(0, 10, (batch_size,))).to(device)
    x_generated = generator(noise, one_hot_fake)

    # generate loss from fake examples
    y_fake = discriminator(x_generated, one_hot_fake)
    loss = loss_fn(y_fake, torch.ones(batch_size, 1).to(device))

    # backprop
    loss.backward()
    generator_optimizer.step()

    return loss.item()

def train(generator, discriminator, dl, loss_fn, generator_optimizer, discriminator_optimizer, hyperparameters):
    generator.train()
    discriminator.train()

    total_discriminator_real_loss = 0
    total_discriminator_fake_loss = 0
    total_discriminator_combined_loss = 0
    total_generator_loss = 0
    batch_bar = tqdm(total=len(dl), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for j, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        for i in range(hyperparameters['num_discriminator_steps']):
            discriminator_combined_loss, discriminator_real_loss, discriminator_fake_loss = train_discriminator_one_step(generator, discriminator, x, y, loss_fn, discriminator_optimizer, hyperparameters['mock_real'])
            total_discriminator_real_loss += discriminator_real_loss
            total_discriminator_fake_loss += discriminator_fake_loss
            total_discriminator_combined_loss += discriminator_combined_loss
        for i in range(hyperparameters['num_generator_steps']):
            generator_loss = train_generator_one_step(generator, discriminator, x, y, loss_fn, generator_optimizer, hyperparameters['mock_real'])
            total_generator_loss += generator_loss
        
        # update progress bar
        batch_bar.set_postfix(
            disc_real_loss="{:.04f}".format(float(total_discriminator_real_loss / (j + 1))),
            disc_fake_loss="{:.04f}".format(float(total_discriminator_fake_loss / (j + 1))),
            disc_combined_loss="{:.04f}".format(float(total_discriminator_combined_loss / (j + 1))),
            gen_loss="{:.04f}".format(float(total_generator_loss / (j + 1))),
        )
        batch_bar.update()
    batch_bar.close()
    print_train_statistics(
        total_discriminator_combined_loss / len(dl),
        total_discriminator_real_loss / len(dl),
        total_discriminator_fake_loss / len(dl),
        total_generator_loss / len(dl),
    )


def val(generator, discriminator, dl, loss_fn, hyperparameters, save_outputs=True):
    # Just doing accuracy of detection for now, later can add something like FID
    generator.eval()
    discriminator.eval()
    
    total_discriminator_real_loss = 0
    total_discriminator_fake_loss = 0
    total_discriminator_combined_loss = 0
    total_generator_loss = 0
    total_real_correct = 0
    total_fake_incorrect = 0
    total_count = 0
    batch_bar = tqdm(total=len(dl), dynamic_ncols=True, leave=False, position=0, desc='Validation')

    batch_size = 0
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        batch_size = x.shape[0]

        # generate examples
        noise = generator.generate_noise(batch_size).to(device)
        one_hot = mnist_to_one_hot(y).to(device)
        if hyperparameters['mock_real']:
            one_hot_fake = one_hot
        else:
            one_hot_fake = mnist_to_one_hot(torch.randint(0, 10, (batch_size,))).to(device)
        x_generated = generator(noise, one_hot_fake)

        # generate loss from real examples
        y_real = discriminator(x, one_hot)
        real_loss = loss_fn(y_real, torch.ones(batch_size, 1).to(device))

        # generate loss from fake examples
        y_fake = discriminator(x_generated, one_hot_fake)
        fake_loss = loss_fn(y_fake, torch.zeros(batch_size, 1).to(device))

        # combine loss
        loss = real_loss + fake_loss

        # calculate generator loss
        generator_loss = loss_fn(y_fake, torch.ones(batch_size, 1).to(device))

        # update statistics
        total_discriminator_real_loss += real_loss.item()
        total_discriminator_fake_loss += fake_loss.item()
        total_discriminator_combined_loss += loss.item()
        total_generator_loss += generator_loss.item()
        total_real_correct += (y_real > 0.5).sum().item()
        total_fake_incorrect += (y_fake > 0.5).sum().item()
        total_count += len(x)

        # update progress bar
        batch_bar.set_postfix(
            disc_real_loss="{:.04f}".format(float(total_discriminator_real_loss / (i + 1))),
            disc_fake_loss="{:.04f}".format(float(total_discriminator_fake_loss / (i + 1))),
            disc_combined_loss="{:.04f}".format(float(total_discriminator_combined_loss / (i + 1))),
            gen_loss="{:.04f}".format(float(total_generator_loss / (i + 1))),
            real_correct="{:.04f}%".format(float(total_real_correct * 100 / total_count)),
            fake_incorrect="{:.04f}%".format(float(total_fake_incorrect * 100 / total_count)),
        )
        batch_bar.update()
    batch_bar.close()
    print_validation_statistics(
        total_discriminator_combined_loss / len(dl),
        total_discriminator_real_loss / len(dl),
        total_discriminator_fake_loss / len(dl),
        total_generator_loss / len(dl),
        total_real_correct * 100 / total_count,
        total_fake_incorrect * 100 / total_count,
    )

    # output generated images to file
    if save_outputs:
        visualize_mnist(generator)


# train
for epoch in range(hyperparameters['epochs']):
    print(
        f"Epoch {epoch+1}/{hyperparameters["epochs"]}, " +
        f"Generator LR: {generator_optimizer.param_groups[0]['lr']:.06f}, " +
        f"Discriminator LR: {discriminator_optimizer.param_groups[0]['lr']:.06f}"
    )

    train(generator, discriminator, train_dl, loss_fn, generator_optimizer, discriminator_optimizer, hyperparameters)
    val(generator, discriminator, val_dl, loss_fn, hyperparameters, save_outputs=True)

    if hyperparameters['lr_scheduler']:
        if not hyperparameters['min_lr'] or generator_optimizer.param_groups[0]['lr'] > hyperparameters['min_lr']:
            generator_lr_scheduler.step()
        if not hyperparameters['min_lr'] or discriminator_optimizer.param_groups[0]['lr'] > hyperparameters['min_lr']:
            discriminator_lr_scheduler.step()

    if epoch % 10 == 9:
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': generator_optimizer.state_dict(),
            'scheduler_state_dict': generator_lr_scheduler.state_dict() if hyperparameters['lr_scheduler'] else None,
        }, os.path.join(hyperparameters['save_directory'], f"epoch_{epoch}_generator.pt"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': discriminator_optimizer.state_dict(),
            'scheduler_state_dict': discriminator_lr_scheduler.state_dict() if hyperparameters['lr_scheduler'] else None,
        }, os.path.join(hyperparameters['save_directory'], f"epoch_{epoch}_discriminator.pt"))
