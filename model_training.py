import os
import torch
import torchvision
from time import time

if __name__ == "__main__":
    # Hyperparameters
    LR = 0.0009
    DATA = "datasets/cifar10/"
    BATCH_SIZE = 256
    CHECKPOINT_PATH = "checkpoint.pth"

    # Data transformations
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)).to('xpu', non_blocking=True),
            torchvision.transforms.RandomHorizontalFlip().to('xpu', non_blocking=True),
            torchvision.transforms.RandomRotation(15).to('xpu', non_blocking=True),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1).to('xpu', non_blocking=True),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).to('xpu', non_blocking=True)
        ]
    )
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=True,  # Set to false if Dataset already downloaded
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    train_len = len(train_loader)

    # Initialize model, criterion, and optimizer
    model = torchvision.models.resnet50().to("xpu", non_blocking=True)
    for param in model.conv1.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, 1000).to('xpu', non_blocking=True)

    criterion = torch.nn.CrossEntropyLoss().to('xpu', non_blocking=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Load checkpoint from the previous training
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Starting from scratch.")

    for i in range(1):
        print(f"Starting epoch {start_epoch + 1}")
        model.train()
        start_time = time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to("xpu", non_blocking=True)
            target = target.to("xpu", non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="xpu"):
                output = model(data)
                loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if (batch_idx + 1) % 10 == 0:
                elapsed_time = time() - start_time
                print(
                    f"Epoch [{start_epoch + 1}], Iteration [{batch_idx + 1}/{train_len}], "
                    f"Loss: {loss.item():.4f}, Elapsed Time: {elapsed_time:.2f} seconds"
                )
        scheduler.step()

        # Save the updated checkpoint after this epoch
        torch.save(
            {
                "epoch": start_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            CHECKPOINT_PATH,
        )
        print(f"Checkpoint saved after epoch {start_epoch + 1}.")
