import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import FilterDataset
from model import CNN
from pytorch_msssim import ssim

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameters
num_epochs = 30
batch_size = 16
learning_rate = 0.001

# define SSIM Loss
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, output, target):
        return 1 - ssim(output, target, data_range=1, size_average=True)

# prepare dataset and data loader
transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = FilterDataset("dataset/train/original2", "dataset/train/filtered", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# initialize model, loss function, and optimizer
model = CNN().to(device)
criterion = SSIMLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        outputs = model(inputs)
        #print("Model output shape:", outputs.shape)

        loss = criterion(outputs, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# save the trained model
torch.save(model.state_dict(), "filter_model.pth")
print("Model training complete and saved as 'filter_model.pth'.")
