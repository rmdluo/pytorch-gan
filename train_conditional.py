# torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from utils import mnist_to_one_hot

# tqdm import
from tqdm import tqdm

# matplotlib import
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
    'num_discriminator_steps': 2,
    'num_generator_steps': 1,
    'epochs': 200,

    # optimizer settings
    'discriminator_optimizer': torch.optim.AdamW,
    'discriminator_optimizer_settings': {
        'lr': 3e-4,
        'betas': (0.5, 0.999),
    },
    'generator_optimizer': torch.optim.AdamW,
    'generator_optimizer_settings': {
        'lr': 5e-4,
        'betas': (0.5, 0.999),
    },

    # lr scheduler settings
    'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
    'lr_scheduler_settings': {
        'T_max': 200,
        'eta_min': 1e-7,
    },
    'min_lr': 0,

    # architecture
    'noise_dim': 100,
    'cond_dim': 10,
    'generator_activation': torch.nn.ReLU,
    'generator_final_activation': torch.nn.Sigmoid,
    'generator_dropout_prob': 0.5,
    'generator_normalize': torch.nn.BatchNorm1d,
    'discriminator_activation': torch.nn.LeakyReLU,
    'discriminator_final_activation': torch.nn.Sigmoid,
    'discriminator_dropout_prob': 0.5,
    'discriminator_normalize': None,

    # example image output settings
    'num_images': 4,
    'save_directory': 'conditional'
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
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())])
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
    hyperparameters['generator_final_activation'],
    hyperparameters['generator_dropout_prob'],
    hyperparameters['generator_normalize'],
)
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
    hyperparameters['discriminator_final_activation'],
    hyperparameters['discriminator_dropout_prob'],
    hyperparameters['discriminator_normalize'],
)
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


# Train/Val loops
def train_discriminator(generator, discriminator, dl, loss_fn, discriminator_optimizer, hyperparameters):
    generator.train()
    discriminator.train()

    for i in range(hyperparameters['num_discriminator_steps']):
        total_real_loss = 0
        total_fake_loss = 0
        total_loss = 0
        batch_bar = tqdm(total=len(dl), dynamic_ncols=True, leave=False, position=0, desc=f'Train Discriminator Step {i+1}')

        for j, (x, y) in enumerate(dl):
            x = x.to(device)
            batch_size = x.shape[0]

            # zero out gradients
            discriminator_optimizer.zero_grad()

            # generate examples
            noise = torch.randn((batch_size, hyperparameters['noise_dim']))
            noise = noise.to(device)
            one_hot = mnist_to_one_hot(y).to(device)
            x_generated = generator(noise, one_hot)

            # generate loss from real examples
            y_real = discriminator(x, one_hot)
            real_loss = loss_fn(y_real, torch.ones(batch_size, 1).to(device))

            # generate loss from fake examples
            y_fake = discriminator(x_generated.detach(), one_hot)
            fake_loss = loss_fn(y_fake, torch.zeros(batch_size, 1).to(device))

            # combine loss
            loss = real_loss + fake_loss

            # backprop
            loss.backward()
            discriminator_optimizer.step()

            # update batch bar
            total_real_loss += real_loss.item()
            total_fake_loss += fake_loss.item()
            total_loss += loss.item()
            batch_bar.set_postfix(
                real_loss="{:.04f}".format(float(total_real_loss / (j + 1))),
                fake_loss="{:.04f}".format(float(total_fake_loss / (j + 1))),
                loss="{:.04f}".format(float(total_loss / (j + 1))),
            )
            batch_bar.update()
        batch_bar.close()
        print(
            f"Train Discriminator Step {i+1}: " +
            f"fake_loss={float(total_fake_loss / len(dl)):.04f}, " +
            f"loss={total_loss / len(dl):.04f}, " +
            f"real_loss=loss={total_real_loss / len(dl):.04f}"
        )

def train_generator(generator, discriminator, dl, loss_fn, generator_optimizer, hyperparameters):
    generator.train()
    discriminator.train()
    
    for i in range(hyperparameters['num_generator_steps']):
        total_loss = 0
        batch_bar = tqdm(total=len(dl), dynamic_ncols=True, leave=False, position=0, desc=f'Train Generator Step {i+1}')

        for j, (x, y) in enumerate(dl):
            x = x.to(device)
            batch_size = x.shape[0]

            # zero out gradients
            generator_optimizer.zero_grad()

            # generate examples
            noise = torch.randn((batch_size, hyperparameters['noise_dim']))
            noise = noise.to(device)
            one_hot = mnist_to_one_hot(y).to(device)
            x_generated = generator(noise, one_hot)

            # generate loss from fake examples
            y_fake = discriminator(x_generated, one_hot)
            loss = loss_fn(y_fake, torch.ones(batch_size, 1).to(device))

            # backprop
            loss.backward()
            generator_optimizer.step()

            # update batch bar
            total_loss += loss.item()
            batch_bar.set_postfix(
                loss="{:.04f}".format(float(total_loss / (j + 1))),
            )
            batch_bar.update()
        batch_bar.close()
        print(f'Train Generator Step {i+1}: loss={total_loss / len(dl):.04f}')

def val(generator, discriminator, dl, loss_fn, hyperparameters, save_outputs=True):
    # Just doing accuracy of detection for now, later can add something like FID
    generator.eval()
    discriminator.eval()
    
    total_loss = 0
    total_acc = 0
    batch_bar = tqdm(total=len(dl), dynamic_ncols=True, leave=False, position=0, desc='Validation')

    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        batch_size = x.shape[0]

        # generate examples
        noise = torch.randn((batch_size, hyperparameters['noise_dim']))
        noise = noise.to(device)
        one_hot = mnist_to_one_hot(y).to(device)
        x_generated = generator(noise, one_hot)

        # generate loss from real examples
        y_real = discriminator(x, one_hot)
        real_loss = loss_fn(y_real, torch.ones(batch_size, 1).to(device))

        # generate loss from fake examples
        y_fake = discriminator(x_generated, one_hot)
        fake_loss = loss_fn(y_fake, torch.zeros(batch_size, 1).to(device))

        # combine loss
        loss = real_loss + fake_loss
        total_loss += loss.item()

        # calculate accuracy
        total_acc += ((y_real > 0.5).sum() + (y_fake <= 0.5).sum()).item() / (2 * batch_size)

        # update progress bar
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            acc="{:.04f}%".format(float(total_acc * 100 / (i + 1)))
        )
        batch_bar.update()
    batch_bar.close()
    print(f'Validation: acc={total_acc * 100 / len(dl):.04f}, loss={total_loss / len(dl):.04f}')

    # output generated images to file
    if save_outputs:
        size = min(batch_size, hyperparameters["num_images"] if hyperparameters["num_images"] else batch_size)
        img_size = int(math.sqrt(x_generated[0].shape[-1]))
        selected_indices = random.sample(range(len(x_generated)), size)
        fig, axes = plt.subplots(1, size)

        for idx, j in enumerate(selected_indices):
            img = torch.clamp(x_generated[j], 0, 255).cpu().detach().numpy().reshape(img_size, img_size)  # Assuming square
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f"Digit: {y[j].item()}", fontsize=10)
            axes[idx].axis('off')
            
        plt.savefig(os.path.join(hyperparameters['save_directory'], f"generated_epoch_{epoch}.png"))
        plt.close()

    return total_loss / len(dl), total_acc * 100 / len(dl)


# train
worst_acc = 100
for epoch in range(hyperparameters['epochs']):
    print(
        f"Epoch {epoch+1}/{hyperparameters["epochs"]}, " +
        f"Generator LR: {generator_optimizer.param_groups[0]['lr']:.06f}, " +
        f"Discriminator LR: {discriminator_optimizer.param_groups[0]['lr']:.06f}"
    )

    train_discriminator(generator, discriminator, train_dl, loss_fn, discriminator_optimizer, hyperparameters)
    val(generator, discriminator, val_dl, loss_fn, hyperparameters, save_outputs=False)
    train_generator(generator, discriminator, train_dl, loss_fn, generator_optimizer, hyperparameters)
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