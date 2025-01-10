from datasets import load_dataset
import torch
from Dataset import load_data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from CNN import MiniVGG_BN


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience,device):
  model.train()
  best_val_loss = float('inf')
  epochs_no_improve = 0
  for epoch in range(epochs):
    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=True)

    running_loss = 0.0
    total_samples = 0
    total_correct = 0

    for inputs, labels in train_loader_tqdm:
      # Set tensor to device
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad() # Make sure gradient is set to zero
      # Forward pass
      outputs = model(inputs)
      _, preds  = torch.max(outputs, dim=1)
      loss = criterion(outputs, labels)#Calculate loss
      # Backward pass
      loss.backward()
      optimizer.step()
      # Calculate running loss and accuracy
      total_correct += (preds == labels).sum().item()
      running_loss += loss.item()*inputs.size(0)
      total_samples +=inputs.size(0)
      train_loader_tqdm.set_postfix(loss=running_loss/total_samples)
    # Calculate avg training loss for this epoch
    avg_train_loss = running_loss / total_samples
    train_acc = total_correct*100/total_samples
    # Calculate Val Loss for this epoch
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    # Display training loss for this epoch and log to wandb
    wandb.log({"epoch":epoch+1 ,"train_loss": avg_train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    print(f"Epoch {epoch+1}/{epochs},Train Loss: {avg_train_loss:.4f} | Train Acc:{train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      epochs_no_improve = 0
      # Save the best model
      torch.save(model.state_dict(), "weight/best_model.pth")
    else:
      epochs_no_improve +=1
      print(f"No improvement in validation loss. Patience counter: {epochs_no_improve}/{patience}")

    if epochs_no_improve >= patience:
      print("Early stopping triggered")
      break

if __name__ == "__main__":
   
   train_loader,val_loader,test_loader = load_data(64)
   device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
   model = MiniVGG_BN().to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr = 0.0005)
   wandb.init(project = "Trash-Classification",
           config = {
               "epochs" : 25,
               "batch_size" :64,
               "learning_rate" : 0.001,
               "architecture" : "CNN",
               "num_classes" : 6
           })
   train_model(model,train_loader,val_loader,optimizer,criterion,25,3,device=device)
   wandb.finish()
   
    