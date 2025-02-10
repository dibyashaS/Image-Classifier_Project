import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Parser for train.py')
    parser.add_argument('data_dir', type=str, default="./flowers/")
    parser.add_argument('--save_dir', type=str, default="./checkpoint.pth")
    parser.add_argument('--arch', type=str, default="vgg16")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

# Model setup
def setup_model(arch, hidden_units, dropout):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, hidden_units)
    model.classifier.append(nn.Dropout(dropout))
    model.classifier.append(nn.LogSoftmax(dim=1))
    return model

# Data loading
def load_data(data_dir):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(30),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_data = datasets.Flowers102(data_dir, transform=transform)
    train_data, val_data = random_split(full_data, [int(0.8 * len(full_data)), len(full_data) - int(0.8 * len(full_data))])
    return DataLoader(train_data, batch_size=64, shuffle=True), DataLoader(val_data, batch_size=64, shuffle=True)

# Training and validation
def train_and_validate(model, train_loader, val_loader, epochs, lr, gpu):
    criterion, optimizer = nn.NLLLoss(), optim.Adam(model.classifier.parameters(), lr)
    if gpu and torch.cuda.is_available():
        model.cuda()
    
    for epoch in range(epochs):
        model.train()
        running_loss = sum(criterion(model(images.cuda() if gpu else images), labels.cuda() if gpu else labels).item()
                           for images, labels in train_loader)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, gpu)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Training Loss: {running_loss/len(train_loader):.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Accuracy: {val_accuracy:.2f}%')

# Validation
def validate_model(model, val_loader, criterion, gpu):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            output = model(images)
            val_loss += criterion(output, labels).item()
            correct += (output.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return val_loss / len(val_loader), 100 * correct / total

# Save model
def save_checkpoint(model, save_dir):
    torch.save({'state_dict': model.state_dict()}, save_dir)

# Main function
def main():
    args = parse_args()
    model = setup_model(args.arch, args.hidden_units, args.dropout)
    train_loader, val_loader = load_data(args.data_dir)
    train_and_validate(model, train_loader, val_loader, args.epochs, args.learning_rate, args.gpu)
    save_checkpoint(model, args.save_dir)

if __name__ == '__main__':
    main()


