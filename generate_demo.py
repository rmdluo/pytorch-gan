import json
import torch
import matplotlib.pyplot as plt
from mlp.models import ConditionalGenerator #, ConditionalDiscriminator
from utils import mnist_to_one_hot

source_folder = "conditional_10"
epoch = 199
config_path = f"{source_folder}/config.json"
weights_generator_path = f"{source_folder}/epoch_{epoch}_generator.pt"
# weights_discriminator_path = "conditional_26/epoch_199_discriminator.pt"

with open(config_path) as f:
    hyperparameters = json.load(f)
    if "ReLU" in hyperparameters['generator_activation']:
        hyperparameters['generator_activation'] = torch.nn.ReLU
    if "Sigmoid" in hyperparameters['generator_final_activation']:
        hyperparameters['generator_final_activation'] = torch.nn.Sigmoid
    if hyperparameters['generator_normalize'] or "BatchNorm1d" in hyperparameters['generator_normalize']:
        hyperparameters['generator_normalize'] = torch.nn.BatchNorm1d
weights_generator = torch.load(weights_generator_path, weights_only=True)['model_state_dict']
new_weights_generator = {}
for key in weights_generator.keys():
    new_key = key.replace(".norm", ".normalize")  # Replace incorrect layer names
    new_weights_generator[new_key] = weights_generator[key]
weights_generator = new_weights_generator
# weights_discriminator = torch.load(weights_discriminator_path, weights_only=True)['model_state_dict']

image_shape = 784
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


### ChatGPT generated
num_samples_per_digit = 10
num_classes = 10

generator.load_state_dict(weights_generator)
generator.eval()

def generate_digits(generator, num_samples_per_digit, num_classes):
    noise_dim = hyperparameters['noise_dim']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    
    noise = torch.randn((num_samples_per_digit * num_classes, noise_dim), device=device)
    labels = torch.arange(num_classes, device=device).repeat_interleave(num_samples_per_digit)
    labels = mnist_to_one_hot(labels).to(device)

    with torch.no_grad():
        generated_images = generator(noise, labels).cpu()
    
    return generated_images.view(-1, 1, 28, 28)  # Reshape to 28x28 if applicable

def plot_generated_images(images, num_samples_per_digit, num_classes, filename="generated_digits.png"):
    fig, axes = plt.subplots(num_classes, num_samples_per_digit, figsize=(10, 10))
    for i in range(num_classes):
        for j in range(num_samples_per_digit):
            axes[i, j].imshow(images[i * num_samples_per_digit + j].squeeze(), cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(filename)

# Generate and display images
generated_images = generate_digits(generator, num_samples_per_digit, num_classes)
plot_generated_images(generated_images, num_samples_per_digit, num_classes, filename=f"{source_folder}_generated_digits.png")