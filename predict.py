import argparse
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import json
import os

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower class")
    parser.add_argument('image_path', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Top K predictions')
    parser.add_argument('--category_names', type=str, help='Category name mapping')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    return parser.parse_args()

# Load the model
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = models.vgg13(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 512)
    model.classifier.append(nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Preprocess the images
def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0)

# Prediction
def predict(image_path, model, top_k=1, gpu=False):
    model.eval()
    image = process_image(image_path)
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    
    with torch.no_grad():
        output = model(image)
    probs, classes = torch.exp(output).topk(top_k)
    return probs.cpu().numpy(), classes.cpu().numpy()

# Load category names
def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)

# Main function
def main():
    args = parse_args()
    model = load_model(args.checkpoint)
    
    if args.gpu and torch.cuda.is_available():
        model.cuda()

    probs, classes = predict(args.image_path, model, top_k=args.top_k, gpu=args.gpu)
    
    if args.category_names:
        category_names = load_category_names(args.category_names)
        classes = [category_names[str(c)] for c in classes[0]]
    
    for i in range(args.top_k):
        print(f"Class: {classes[i]}, Probability: {probs[0][i]*100:.2f}%")

if __name__ == '__main__':
    main()