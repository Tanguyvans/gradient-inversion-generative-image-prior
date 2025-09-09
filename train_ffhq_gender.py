#!/usr/bin/env python3
"""
Training script for ResNet on FFHQ dataset with gender classification.
This script trains a ResNet model to classify gender from FFHQ face images.

Usage:
    python train_ffhq_gender.py --data_path ./ffhq_dataset --epochs 50 --model ResNet18
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import json
import argparse
import time
import datetime
import numpy as np
from PIL import Image
from collections import defaultdict

import inversefed

class FFHQGenderDataset(Dataset):
    """FFHQ Dataset with gender labels from JSON metadata."""
    
    def __init__(self, root_dir, json_dir, transform=None, split='train', train_ratio=0.8):
        self.root_dir = root_dir
        self.json_dir = json_dir
        self.transform = transform
        self.data = []
        
        # Load all image-label pairs
        image_files = []
        json_files = []
        
        # Get all PNG files in class0 directory
        class_dir = os.path.join(root_dir, 'class0')
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            image_files.sort()
        
        # Get corresponding JSON files
        if os.path.exists(json_dir):
            json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            json_files.sort()
        
        print(f"Found {len(image_files)} images and {len(json_files)} JSON files")
        
        # Process each image-json pair
        for img_file in image_files:
            # Extract ID from filename (e.g., "00000.png" -> "00000")
            img_id = os.path.splitext(img_file)[0]
            json_file = f"{img_id}.json"
            
            img_path = os.path.join(class_dir, img_file)
            json_path = os.path.join(json_dir, json_file)
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract gender from first face in JSON
                    if len(metadata) > 0 and 'faceAttributes' in metadata[0]:
                        gender = metadata[0]['faceAttributes']['gender']
                        label = 1 if gender.lower() == 'male' else 0  # 0: female, 1: male
                        
                        self.data.append({
                            'image_path': img_path,
                            'label': label,
                            'gender': gender,
                            'img_id': img_id
                        })
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
        
        print(f"Successfully loaded {len(self.data)} samples")
        
        # Count gender distribution
        gender_counts = defaultdict(int)
        for item in self.data:
            gender_counts[item['gender']] += 1
        print(f"Gender distribution: {dict(gender_counts)}")
        
        # Split into train/val
        np.random.seed(42)
        indices = np.random.permutation(len(self.data))
        split_idx = int(len(self.data) * train_ratio)
        
        if split == 'train':
            self.data = [self.data[i] for i in indices[:split_idx]]
        else:  # validation
            self.data = [self.data[i] for i in indices[split_idx:]]
        
        print(f"{split.capitalize()} set size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']

def get_transforms(image_size=128):
    """Get data transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train ResNet on FFHQ for gender classification')
    parser.add_argument('--data_path', default='./ffhq_dataset', help='Path to FFHQ dataset')
    parser.add_argument('--json_path', default='./ffhq_json', help='Path to FFHQ JSON metadata')
    parser.add_argument('--model', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='Model architecture')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--image_size', default=128, type=int, help='Image size')
    parser.add_argument('--save_path', default='./models', help='Path to save model')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--single_batch', action='store_true', help='Train on only a single batch of images')
    parser.add_argument('--target_ids', nargs='+', type=int, help='Specific image IDs to use for training (requires --single_batch)')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create datasets
    train_transform, val_transform = get_transforms(args.image_size)
    
    if args.single_batch:
        # Create single batch dataset
        full_dataset = FFHQGenderDataset(
            root_dir=args.data_path,
            json_dir=args.json_path,
            transform=train_transform
        )
        
        if args.target_ids:
            # Use specific target IDs
            target_ids = args.target_ids
            batch_size = len(target_ids)
        else:
            # Use first batch_size images
            target_ids = list(range(min(args.batch_size, len(full_dataset))))
            batch_size = len(target_ids)
        
        print(f"Single batch mode: using {batch_size} images with IDs: {target_ids}")
        
        # Create single batch data
        batch_data = []
        batch_labels = []
        for target_id in target_ids:
            if target_id < len(full_dataset):
                img, label = full_dataset[target_id]
                batch_data.append(img)
                batch_labels.append(label)
        
        batch_data = torch.stack(batch_data)
        batch_labels = torch.tensor(batch_labels)
        
        # Create simple single-batch dataloader
        class SingleBatchLoader:
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            
            def __iter__(self):
                yield self.data, self.labels
            
            def __len__(self):
                return 1
        
        train_loader = SingleBatchLoader(batch_data, batch_labels)
        val_loader = train_loader  # Same data for validation in single batch mode
        
        print(f"Single batch training: {len(batch_data)} samples")
        
    else:
        # Full dataset training (original behavior)
        train_dataset = FFHQGenderDataset(
            root_dir=args.data_path,
            json_dir=args.json_path,
            transform=train_transform,
            split='train'
        )
        
        val_dataset = FFHQGenderDataset(
            root_dir=args.data_path,
            json_dir=args.json_path,
            transform=val_transform,
            split='val'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4 if device.type == 'cuda' else 2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4 if device.type == 'cuda' else 2
        )
        
        print(f"Full dataset training:")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Create model using inversefed
    model, model_seed = inversefed.construct_model(args.model, num_classes=2, num_channels=3)
    model.to(device)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_path, exist_ok=True)
            model_path = os.path.join(args.save_path, f'{args.model}_FFHQ_gender_best.pt')
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Training time: {datetime.timedelta(seconds=training_time)}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(args.save_path, f'{args.model}_FFHQ_gender_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr
    }
    
    import pickle
    history_path = os.path.join(args.save_path, f'{args.model}_FFHQ_gender_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to: {history_path}")

if __name__ == '__main__':
    main()