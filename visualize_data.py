import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to('xpu'),
            torchvision.transforms.RandomHorizontalFlip().to('xpu'),
            torchvision.transforms.RandomRotation(15, interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to('xpu'),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1).to('xpu'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).to('xpu')
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root='datasets/cifar10',
        train = True,
        transform=transform,
        download=False
    )

    loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
    )

        # Helper Function to Denormalize Image
    def denormalize_image(image, mean, std):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        image = image * std + mean  # Denormalize
        return image.clamp(0, 1)  # Clamp values to [0, 1] for display

    # Visualize a Batch of Images Using imshow
    classes = dataset.classes  # Class names
    data_iter = iter(loader)
    images, labels = next(data_iter)  # Get a batch of images and labels

    # Plot Each Image
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        img = denormalize_image(images[i], mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for Matplotlib

        plt.subplot(4, 4, i+1)  # Create a 4x4 grid
        plt.imshow(img)
        plt.title(classes[labels[i].item()])
        plt.axis('off')

    plt.tight_layout()
    plt.show()