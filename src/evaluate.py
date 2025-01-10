
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

from torch.utils.data import random_split
import torch
from Dataset import load_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import wandb
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report
from CNN import MiniVGG_BN

def final_eval(model, test_loader, criterion, device, best_model_path):
    # Load the best model weights
    model.load_state_dict(torch.load(best_model_path, weights_only=True,map_location=device ))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            # Move inputs and labels to the correct device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)  # Get the predicted class

            # Accumulate loss and accuracy
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Collect all predictions and labels for detailed analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    plt.figure(figsize=(9, 9))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    # Log image to wandb
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.close()
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds))
    return avg_loss, accuracy, all_preds, all_labels



if __name__ == "__main__":
    model = MiniVGG_BN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project = "Trash-Classification"
           )
    _,_,test_loader = load_data(64)# use 64 as batch size
    avg_loss, accuracy, all_preds, all_labels = final_eval(
        model=model,
        test_loader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        best_model_path="weight/best_model.pth"
    )
    print(f"Test Accuracy: {(accuracy*100):.2f}%, Test Loss: {avg_loss:.2f}")   
    wandb.finish()
    

    


   

