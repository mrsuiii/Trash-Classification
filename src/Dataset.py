from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import random_split

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
    
def load_data(num_batch = 128):
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    dataset = load_dataset("garythung/trashnet",split = 'train')
    tensor_dataset = HFDatasetWrapper(dataset,transform)
    train_size = int(0.75 * len(tensor_dataset))
    val_size = int(0.15*len(tensor_dataset))
    test_size = len(tensor_dataset) - train_size - val_size
    train_dataset, val_dataset,test_dataset = random_split(tensor_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True,num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=num_batch, shuffle=True,num_workers =0)
    test_loader =DataLoader(test_dataset, batch_size = num_batch,shuffle = True,num_workers = 0)

    return train_loader,val_loader,test_loader