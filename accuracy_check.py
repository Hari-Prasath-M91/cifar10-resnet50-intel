import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import os

if __name__ == "__main__":
    # Initialize model
    model = torchvision.models.resnet50()
    model = model.to("xpu")

    # Hyperparameters
    DATA = "datasets/cifar10/"
    BATCH_SIZE = 128
    CHECKPOINT_PATH = "checkpoint.pth"

    # Data transformations
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load the test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=False,
        transform=transform,
        download=False,
    )
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,  # Enable multiprocessing for data loading
            pin_memory=True,
        )

    # Load checkpoint from the previous training
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Starting from scratch.")

    # Model evaluation
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to compute gradients for evaluation
        for data, target in test_loader:
            data = data.to("xpu")
            target = target.to("xpu")
            output = model(data)
            _, predicted = torch.max(output, 1)  # Get predicted class labels
            all_preds.extend(predicted.cpu().numpy())  # Convert to CPU and store predictions
            all_labels.extend(target.cpu().numpy())  # Convert to CPU and store true labels

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculate additional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')  # For multi-class, use 'weighted'
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
