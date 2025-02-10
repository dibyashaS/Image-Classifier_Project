import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import futility 
import fmodel

# Argument parsing
parser = argparse.ArgumentParser(description='Parser for train.py')
parser.add_argument('data_dir', type=str, default="./flowers/")
parser.add_argument('--save_dir', type=str, default="./checkpoint.pth")
parser.add_argument('--arch', type=str, default="vgg16")
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--gpu', action='store_true', help="Enable GPU acceleration if available")

args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

def main():
    trainloader, validloader, testloader, train_data = futility.load_data(args.data_dir)
    model, criterion = fmodel.setup_network(args.arch, args.dropout, args.hidden_units, args.learning_rate, args.gpu)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Move model to GPU if enabled
    model.to(device)

    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("--Training starting--")

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1

            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step() #Implemented optimizer step

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()

    # Save the trained model checkpoint
    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'structure': args.arch,
        'hidden_units': args.hidden_units,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'no_of_epochs': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }, args.save_dir)

    print("Saved checkpoint!")

if __name__ == "__main__":
    main()

