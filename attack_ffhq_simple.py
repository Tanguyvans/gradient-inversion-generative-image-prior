#!/usr/bin/env python3
"""
Simplified gradient inversion attack script for FFHQ gender classification models.
This script uses the original inversefed approach more closely.
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from PIL import Image
import inversefed

def load_ffhq_sample(data_path, json_path, target_id, image_size=128):
    """Load a single FFHQ sample with gender label."""
    # Load image
    img_path = os.path.join(data_path, 'class0', f'{target_id:05d}.png')
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Load JSON metadata
    json_file = os.path.join(json_path, f'{target_id:05d}.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        gender = metadata[0]['faceAttributes']['gender']
        label = 1 if gender.lower() == 'male' else 0
    else:
        print(f"Warning: No metadata for image {target_id}, assuming female")
        gender = 'female'
        label = 0
    
    # Load and preprocess image
    image = Image.open(img_path).convert('RGB')
    
    # Simple transform matching FFHQ preprocessing
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image)
    
    return image_tensor, label, gender

def create_simple_loss():
    """Create a simple cross-entropy loss compatible with inversefed."""
    class SimpleLoss:
        def __init__(self):
            self.loss_fn = nn.CrossEntropyLoss()
        
        def __call__(self, outputs, targets):
            loss = self.loss_fn(outputs, targets)
            return loss, None, None
        
        def metric(self, outputs=None, targets=None):
            if outputs is not None and targets is not None:
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == targets).float().mean()
                return correct, 'acc', '6.4f'
            else:
                return None, 'acc', '6.4f'
    
    return SimpleLoss()

def main():
    parser = argparse.ArgumentParser(description='Simple gradient inversion attack on FFHQ')
    parser.add_argument('--model_path', default='./models/ResNet18_FFHQ_gender_best.pt', help='Path to trained model (.pt format)')
    parser.add_argument('--data_path', default='./ffhq_dataset', help='Path to FFHQ dataset')
    parser.add_argument('--json_path', default='./ffhq_json', help='Path to FFHQ JSON metadata')
    parser.add_argument('--target_id', default=0, type=int, help='Target image ID')
    parser.add_argument('--max_iterations', default=1000, type=int, help='Attack iterations')
    parser.add_argument('--lr', default=0.1, type=float, help='Attack learning rate')
    parser.add_argument('--tv', default=1e-6, type=float, help='Total variation weight')
    parser.add_argument('--cost_fn', default='sim', help='Cost function')
    parser.add_argument('--restarts', default=1, type=int, help='Number of restarts')
    parser.add_argument('--model_name', default='ResNet18', help='Model architecture')
    parser.add_argument('--image_size', default=128, type=int, help='Image size')
    parser.add_argument('--device', default='auto', help='Device')
    parser.add_argument('--save_images', action='store_true', help='Save images')
    parser.add_argument('--results_path', default='./stylegan_attack_results', help='Results path')
    parser.add_argument('--use_stylegan', action='store_true', help='Use StyleGAN2 as generative prior')
    parser.add_argument('--target_ids', nargs='+', type=int, help='Multiple target IDs to attack')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    setup = dict(dtype=torch.float, device=device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model, _ = inversefed.construct_model(args.model_name, num_classes=2, num_channels=3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(**setup)
    model.eval()
    print("Model loaded successfully")
    
    # Determine target IDs to attack
    if args.target_ids:
        target_ids = args.target_ids
    else:
        target_ids = [args.target_id]
    
    print(f"Will attack {len(target_ids)} images: {target_ids}")
    
    # Attack each target
    results = []
    for i, target_id in enumerate(target_ids):
        print(f"\n{'='*50}")
        print(f"ATTACKING IMAGE {i+1}/{len(target_ids)}: ID {target_id}")
        print(f"{'='*50}")
        
        # Load target image
        try:
            target_image, target_label, gender = load_ffhq_sample(
                args.data_path, args.json_path, target_id, args.image_size
            )
            target_image = target_image.unsqueeze(0).to(**setup)
            target_label = torch.tensor([target_label], device=device)
            
            print(f"Target gender: {gender}")
            print(f"Target label: {target_label.item()} ({'male' if target_label.item() == 1 else 'female'})")
            
        except FileNotFoundError as e:
            print(f"Skipping target ID {target_id}: {e}")
            continue
        
        # Test model prediction
        with torch.no_grad():
            pred_logits = model(target_image)
            pred_class = torch.argmax(pred_logits, dim=1).item()
            pred_confidence = torch.softmax(pred_logits, dim=1).max().item()
        
        print(f"Model prediction: {pred_class} ({'male' if pred_class == 1 else 'female'})")
        print(f"Confidence: {pred_confidence:.3f}")
        
        # Create loss function
        loss_fn = create_simple_loss()
        
        # Compute gradients
        model.zero_grad()
        target_loss, _, _ = loss_fn(model(target_image), target_label)
        input_gradients = torch.autograd.grad(target_loss, model.parameters())
        input_gradients = [grad.detach() for grad in input_gradients]
        
        print(f"Target loss: {target_loss.item():.4f}")
        
        # Setup data normalization (FFHQ uses different normalization)
        dm = torch.tensor([0.5, 0.5, 0.5], **setup)[:, None, None]  # FFHQ mean
        ds = torch.tensor([0.5, 0.5, 0.5], **setup)[:, None, None]  # FFHQ std
        
        # Attack config
        if args.use_stylegan:
            config = dict(
                signed=True,
                cost_fn=args.cost_fn,
                indices='def',
                weights='equal',
                lr=args.lr,
                optim='adam',
                restarts=args.restarts,
                max_iterations=args.max_iterations,
                total_variation=args.tv,
                bn_stat=0,
                image_norm=0,
                z_norm=0,
                group_lazy=0,
                init='randn',
                lr_decay=True,
                dataset='FFHQ',
                generative_model='stylegan2',  # Use StyleGAN2!
                gen_dataset='FFHQ',
                giml=False,
                gias=False
            )
            method_name = "StyleGAN2"
        else:
            config = dict(
                signed=True,  # Use signed gradients
                cost_fn=args.cost_fn,
                indices='def',
                weights='equal',
                lr=args.lr,
                optim='adam',
                restarts=args.restarts,
                max_iterations=args.max_iterations,
                total_variation=args.tv,
                bn_stat=0,  # Disable BatchNorm statistics
                image_norm=0,
                z_norm=0,
                group_lazy=0,
                init='randn',
                lr_decay=True,
                dataset='FFHQ',
                generative_model='',
                gen_dataset='',
                giml=False,
                gias=False
            )
            method_name = "Direct"
        
        print(f"Using {method_name} reconstruction method")
        
        # Perform attack
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
        
        try:
            reconstructed, stats = rec_machine.reconstruct(
                [input_gradients], 
                target_label, 
                img_shape=target_image.shape[1:], 
                dryrun=False
            )
            
            # Compute metrics
            target_denorm = torch.clamp(target_image * ds + dm, 0, 1)
            reconstructed_denorm = torch.clamp(reconstructed * ds + dm, 0, 1)
            
            mse = (reconstructed_denorm - target_denorm).pow(2).mean().item()
            psnr = inversefed.metrics.psnr(reconstructed_denorm, target_denorm, factor=1)
            
            print(f"\nAttack completed!")
            
            # Handle stats properly (it's a defaultdict)
            if hasattr(stats, 'get'):
                recon_loss = stats.get('opt', 0.0)
            elif isinstance(stats, dict) and 'opt' in stats:
                recon_loss = stats['opt']
            else:
                recon_loss = float(stats) if stats is not None else 0.0
                
            print(f"Reconstruction loss: {recon_loss:.4f}")
            print(f"MSE: {mse:.6f}")
            print(f"PSNR: {psnr:.2f} dB")
            
            # Store results
            results.append({
                'target_id': target_id,
                'gender': gender,
                'pred_class': pred_class,
                'pred_confidence': pred_confidence,
                'mse': mse,
                'psnr': psnr,
                'recon_loss': recon_loss
            })
            
            # Save images
            if args.save_images:
                os.makedirs(args.results_path, exist_ok=True)
                
                # Original
                torchvision.utils.save_image(
                    target_denorm,
                    os.path.join(args.results_path, f'original_{target_id}_{gender}.png')
                )
                
                # Reconstructed
                torchvision.utils.save_image(
                    reconstructed_denorm,
                    os.path.join(args.results_path, f'reconstructed_{target_id}_{gender}.png')
                )
                
                # Side-by-side
                comparison = torch.cat([target_denorm, reconstructed_denorm], dim=3)
                torchvision.utils.save_image(
                    comparison,
                    os.path.join(args.results_path, f'comparison_{target_id}_{gender}.png')
                )
                
                print(f"Images saved to {args.results_path}")
            
        except Exception as e:
            print(f"Attack failed: {e}")
            results.append({
                'target_id': target_id,
                'gender': gender,
                'error': str(e)
            })
            import traceback
            traceback.print_exc()
    
    # Print summary
    if results:
        print(f"\n{'='*50}")
        print("ATTACK SUMMARY")
        print(f"{'='*50}")
        
        successful_results = [r for r in results if 'psnr' in r]
        failed_results = [r for r in results if 'error' in r]
        
        print(f"Total attacks: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            psnr_values = [r['psnr'] for r in successful_results]
            mse_values = [r['mse'] for r in successful_results]
            
            print(f"Average PSNR: {np.mean(psnr_values):.2f} ± {np.std(psnr_values):.2f} dB")
            print(f"Average MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
            print(f"Best PSNR: {np.max(psnr_values):.2f} dB")
            
            # Per-gender analysis
            male_results = [r for r in successful_results if r['gender'] == 'male']
            female_results = [r for r in successful_results if r['gender'] == 'female']
            
            if male_results:
                male_psnr = [r['psnr'] for r in male_results]
                print(f"Male faces PSNR: {np.mean(male_psnr):.2f} ± {np.std(male_psnr):.2f} dB ({len(male_results)} samples)")
            
            if female_results:
                female_psnr = [r['psnr'] for r in female_results]
                print(f"Female faces PSNR: {np.mean(female_psnr):.2f} ± {np.std(female_psnr):.2f} dB ({len(female_results)} samples)")

if __name__ == '__main__':
    main()