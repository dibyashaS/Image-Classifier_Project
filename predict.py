import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json

# Parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict image class')
    parser.add_argument('image_path', type=str, help='Path to the image to be predicted')
    parser.add_argument('checkpoint', type=str, help='Path to the saved checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

# Load model
def load_model(checkpoint_path, gpu):
    checkpoint = torch.load(checkpoint_path)
    
    # Retrieve the architecture from the checkpoint
    arch = checkpoint['arch']  # This assumes the architecture is saved in the checkpoint
    
    # Dynamically load the model based on the saved architecture
    model = getattr(models, arch)(pretrained=True)  # Use getattr to dynamically load the model
    
    # Replace the model's classifier with the saved one
    model.classifier = checkpoint['state_dict']['classifier']  # Assuming classifier is saved in the checkpoint
    model.load_state_dict(checkpoint['state_dict']['state_dict'])  # Load model weights
    
    # Restore other information (e.g., class_to_idx)
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Set device
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    return model

# Process image
def process_image(image_path):
    image = Image.open(image_path)
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    return image.unsqueeze(0)

# Predict image class
def predict(image_path, model, top_k, category_names, gpu):
    image = process_image(image_path)
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    probs, classes = torch.exp(output).topk(top_k)
    
    # Load category names mapping
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Map indices to actual class labels using model.class_to_idx
    class_idx = [model.class_to_idx[str(c)] for c in classes.cpu().numpy()[0]]
    
    # Map indices to category names
    class_names = [cat_to_name[idx] for idx in class_idx]
    return class_names, probs.cpu().numpy()[0]

# Main function
def main():
    args = get_args()
    model = load_model(args.checkpoint, args.gpu)
    class_names, probs = predict(args.image_path, model, args.top_k, args.category_names, args.gpu)
    
    print(f"Top {args.top_k} Predictions:")
    for name, prob in zip(class_names, probs):
        print(f"{name}: {prob:.4f}")

if __name__ == '__main__':
    main()
