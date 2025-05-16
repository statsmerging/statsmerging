import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm



# -------------------------------
# Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "/home/brcao/Repos/merge_model/Datasets/mm/cifar10/"

os.makedirs(data_root)
save_path = os.path.join(data_root, "resnet50_finetuned.pt")
num_epochs = 20
batch_size = 128
lr = 1e-3

# -------------------------------
# Data Preparation
# -------------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                             download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                            download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -------------------------------
# Model Definition
# -------------------------------
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# -------------------------------
# Training Loop
# -------------------------------
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc = evaluate(model, test_loader)
    scheduler.step(val_acc)

    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model with ACC: {best_acc * 100:.2f}%")

print(f"\n Training complete. Best Accuracy: {best_acc * 100:.2f}%")
print(f"Best checkpoint saved at: {save_path}")
