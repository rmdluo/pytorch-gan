import torch
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mnist_to_one_hot(digits):
    one_hot = torch.zeros((len(digits), 10))
    one_hot[torch.arange(len(digits)), digits] = 1
    return one_hot

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

def visualize_mnist(generator, save_path, device):
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

    plt.savefig(save_path)
    plt.close()