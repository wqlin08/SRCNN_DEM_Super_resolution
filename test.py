import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
import argparse
import rasterio
from rasterio.plot import show
from tqdm import tqdm
import time
import glob
from pathlib import Path
import cv2
from PIL import Image
import torch.nn.functional as F
import csv

# Import custom modules
from train_srcnn import SRCNN, DEMDataset, HR_SIZE, LR_SIZE

# Set directories
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)
tif_dir = os.path.join(result_dir, "tif_results")
os.makedirs(tif_dir, exist_ok=True)

# Set log files
test_log_file = os.path.join(log_dir, "test_results.txt")
test_csv_file = os.path.join(log_dir, "test_metrics.csv")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to write logs
def write_log(message):
    with open(test_log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# Functions to calculate metrics
def calculate_psnr(img1, img2, data_range=None):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    if data_range is None:
        data_range = max(img1.max() - img1.min(), 1e-6)
    return 20 * np.log10(data_range / (np.sqrt(mse) + 1e-10))

def calculate_ssim(img1, img2, data_range=None):
    """Calculate Structural Similarity Index (SSIM)"""
    if data_range is None:
        data_range = max(img1.max() - img1.min(), 1e-6)
    try:
        return ssim(img1, img2, data_range=data_range)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error (MAE)"""
    return np.mean(np.abs(img1 - img2))

def calculate_rmse(img1, img2):
    """Calculate Root Mean Squared Error (RMSE)"""
    mse = np.mean((img1 - img2) ** 2)
    return np.sqrt(mse)

def save_as_tif(data, output_path, original_img_path=None):
    """Save as GeoTIFF format"""
    try:
        # If original image is provided, retain its geospatial information
        if original_img_path and original_img_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(original_img_path) as src:
                # Get metadata
                profile = src.profile.copy()
                # Update dimensions
                profile.update(
                    width=data.shape[1],
                    height=data.shape[0],
                    count=1,
                    dtype=data.dtype
                )
                # Write data
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                print(f"Saved GeoTIFF (with geospatial reference): {output_path}")
        else:
            # If no geospatial info, save as regular TIFF
            # Ensure data is within valid range
            data_min, data_max = np.min(data), np.max(data)
            if data.dtype == np.float32 or data.dtype == np.float64:
                # Save as 32-bit float TIFF
                im = Image.fromarray(data)
                im.save(output_path)
            else:
                # Save as 16-bit integer TIFF
                data_scaled = ((data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
                im = Image.fromarray(data_scaled)
                im.save(output_path)
            print(f"Saved TIFF (without geospatial reference): {output_path}")
    except Exception as e:
        print(f"Error saving TIFF: {e}")

def visualize_triptych(lr_img, sr_img, hr_img, output_path, title=None):
    """Generate and save triptych visualization with improved colormap display, colormap below images"""
    # Get common value range for consistent colormap
    min_val = min(np.min(lr_img), np.min(sr_img), np.min(hr_img))
    max_val = max(np.max(lr_img), np.max(sr_img), np.max(hr_img))
    
    # Use terrain colormap
    cmap = 'terrain'
    
    # Create figure with space for colormap
    fig = plt.figure(figsize=(15, 6.5))
    
    # Define subplot grid, leaving space for colormap
    gs = fig.add_gridspec(2, 3, height_ratios=[12, 1], hspace=0.05)
    
    # Create axes for three main images
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot three images
    # Note: LR image retains original size (16x16), no upsampling
    im1 = ax1.imshow(lr_img, cmap=cmap, vmin=min_val, vmax=max_val, interpolation='nearest')
    ax1.set_title(f'Low Resolution ({LR_SIZE}x{LR_SIZE})')
    ax1.axis('off')
    
    im2 = ax2.imshow(sr_img, cmap=cmap, vmin=min_val, vmax=max_val)
    ax2.set_title('SRCNN Super Resolution')
    ax2.axis('off')
    
    im3 = ax3.imshow(hr_img, cmap=cmap, vmin=min_val, vmax=max_val)
    ax3.set_title(f'High Resolution (Ground Truth, {HR_SIZE}x{HR_SIZE})')
    ax3.axis('off')
    
    # Add colormap below images, spanning all three subplots
    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(im3, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Elevation')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")

def test_model(model_path, test_data_dir, num_samples=5, bicubic_compare=True, save_tiff=True):
    """Test the model and calculate evaluation metrics"""
    # Create result directories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(tif_dir, exist_ok=True)
    
    # Clear previous logs
    with open(test_log_file, 'w', encoding='utf-8') as f:
        f.write("Test start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("-" * 100 + "\n")
        f.write("Sample ID | Filename | SRCNN PSNR | SRCNN SSIM | SRCNN MAE | SRCNN RMSE | Bicubic PSNR | Bicubic SSIM | Bicubic MAE | Bicubic RMSE\n")
        f.write("-" * 100 + "\n")
    
    # Create CSV file
    with open(test_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample ID', 'Filename', 'SRCNN PSNR', 'SRCNN SSIM', 'SRCNN MAE', 'SRCNN RMSE', 
                         'Bicubic PSNR', 'Bicubic SSIM', 'Bicubic MAE', 'Bicubic RMSE',
                         'Min Value', 'Max Value', 'Value Range'])
    
    # Load model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {model_path}")
    
    # Load test data
    test_dataset = DEMDataset(test_data_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluation metrics
    srcnn_metrics = {'psnr': [], 'ssim': [], 'mae': [], 'rmse': []}
    bicubic_metrics = {'psnr': [], 'ssim': [], 'mae': [], 'rmse': []}
    
    # Visualize some samples
    samples_to_visualize = min(num_samples, len(test_dataset))
    visualize_indices = np.random.choice(len(test_dataset), samples_to_visualize, replace=False)
    
    # Get list of test files
    test_files = []
    for file in os.listdir(test_data_dir):
        if file.endswith('.tif'):
            test_files.append(os.path.join(test_data_dir, file))
    
    # Use tqdm progress bar
    for idx, (lr_img, hr_img, range_info) in enumerate(tqdm(test_loader, desc="Processing test images")):
        file_path = test_files[idx] if idx < len(test_files) else None
        file_name = os.path.basename(file_path) if file_path else f"sample_{idx}"
        
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        
        # Upsample to original size for super-resolution processing (only for model input)
        lr_upscale = nn.functional.interpolate(lr_img, size=(HR_SIZE, HR_SIZE), mode='bicubic', align_corners=False)
        
        with torch.no_grad():
            sr_img = model(lr_upscale)
        
        # Move to CPU for metric calculation
        sr_np = sr_img.cpu().numpy or numpy.ndarray object.().squeeze()
        hr_np = hr_img.cpu().numpy().squeeze()
        lr_np = lr_img.cpu().numpy().squeeze()  # Keep LR original size (16x16)
        lr_bicubic_np = lr_upscale.cpu().numpy().squeeze()
        
        # Ensure data is within valid range
        sr_np = np.clip(sr_np, 0, 1)
        hr_np = np.clip(hr_np, 0, 1)
        lr_bicubic_np = np.clip(lr_bicubic_np, 0, 1)
        
        # Get original value range
        min_val, max_val = range_info[0].numpy()
        value_range = max_val - min_val
        
        # Restore normalized images to original scale
        sr_np_original = sr_np * value_range + min_val
        hr_np_original = hr_np * value_range + min_val
        lr_np_original = lr_np * value_range + min_val  # Original low-resolution image
        lr_bicubic_original = lr_bicubic_np * value_range + min_val
        
        # Calculate metrics using original skimage and sklearn functions (on original scale)
        # SRCNN metrics
        srcnn_psnr = psnr(hr_np_original, sr_np_original, data_range=value_range)
        srcnn_ssim = ssim(hr_np_original, sr_np_original, data_range=value_range)
        srcnn_mae = mean_absolute_error(hr_np_original, sr_np_original)
        srcnn_rmse = np.sqrt(mean_squared_error(hr_np_original, sr_np_original))
        
        # Collect SRCNN metrics
        srcnn_metrics['psnr'].append(srcnn_psnr)
        srcnn_metrics['ssim'].append(srcnn_ssim)
        srcnn_metrics['mae'].append(srcnn_mae)
        srcnn_metrics['rmse'].append(srcnn_rmse)
        
        # Bicubic interpolation metrics
        bicubic_psnr = psnr(hr_np_original, lr_bicubic_original, data_range=value_range)
        bicubic_ssim = ssim(hr_np_original, lr_bicubic_original, data_range=value_range)
        bicubic_mae = mean_absolute_error(hr_np_original, lr_bicubic_original)
        bicubic_rmse = np.sqrt(mean_squared_error(hr_np_original, lr_bicubic_original))
        
        # Collect bicubic metrics
        bicubic_metrics['psnr'].append(bicubic_psnr)
        bicubic_metrics['ssim'].append(bicubic_ssim)
        bicubic_metrics['mae'].append(bicubic_mae)
        bicubic_metrics['rmse'].append(bicubic_rmse)
        
        # Log results for each sample
        log_message = f"{idx} | {file_name} | {srcnn_psnr:.4f} | {srcnn_ssim:.4f} | {srcnn_mae:.6f} | {srcnn_rmse:.6f} | {bicubic_psnr:.4f} | {bicubic_ssim:.4f} | {bicubic_mae:.6f} | {bicubic_rmse:.6f}"
        write_log(log_message)
        
        # Write results to CSV
        with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([idx, file_name, srcnn_psnr, srcnn_ssim, srcnn_mae, srcnn_rmse, 
                             bicubic_psnr, bicubic_ssim, bicubic_mae, bicubic_rmse,
                             min_val, max_val, value_range])
        
        # Save results as TIF files
        if save_tiff:
            base_name = os.path.splitext(file_name)[0]
            # Save SRCNN result
            sr_tif_path = os.path.join(tif_dir, f"{base_name}_srcnn.tif")
            save_as_tif(sr_np_original, sr_tif_path, file_path)
            
            # # Save Bicubic result
            # if bicubic_compare:
            #     bicubic_tif_path = os.path.join(tif_dir, f"{base_name}_bicubic.tif")
            #     save_as_tif(lr_bicubic_original, bicubic_tif_path, file_path)
        
        # Visualize selected samples
        if idx in visualize_indices:
            # Save visualization result - use original low-resolution image, no upsampling
            vis_title = f"Sample {idx}: {file_name}\nSRCNN: PSNR={srcnn_psnr:.2f}, SSIM={srcnn_ssim:.4f}, MAE={srcnn_mae:.2f}, RMSE={srcnn_rmse:.2f}"
            save_path = os.path.join(result_dir, f'sample_{idx}_{base_name}_compare.png')
            visualize_triptych(lr_np_original, sr_np_original, hr_np_original, save_path, vis_title)
            
            print(f"Sample {idx}: PSNR={srcnn_psnr:.4f}, SSIM={srcnn_ssim:.4f}, MAE={srcnn_mae:.6f}, RMSE={srcnn_rmse:.6f}")
            print(f"Value range: Min={min_val:.2f}, Max={max_val:.2f}, Range={value_range:.2f}")
    
    # Calculate average metrics
    avg_srcnn_psnr = np.mean(srcnn_metrics['psnr'])
    avg_srcnn_ssim = np.mean(srcnn_metrics['ssim'])
    avg_srcnn_mae = np.mean(srcnn_metrics['mae'])
    avg_srcnn_rmse = np.mean(srcnn_metrics['rmse'])
    
    avg_bicubic_psnr = np.mean(bicubic_metrics['psnr'])
    avg_bicubic_ssim = np.mean(bicubic_metrics['ssim'])
    avg_bicubic_mae = np.mean(bicubic_metrics['mae'])
    avg_bicubic_rmse = np.mean(bicubic_metrics['rmse'])
    
    # Log average metrics
    write_log("-" * 100)
    write_log(f"Average | | {avg_srcnn_psnr:.4f} | {avg_srcnn_ssim:.4f} | {avg_srcnn_mae:.6f} | {avg_srcnn_rmse:.6f} | {avg_bicubic_psnr:.4f} | {avg_bicubic_ssim:.4f} | {avg_bicubic_mae:.6f} | {avg_bicubic_rmse:.6f}")
    write_log("-" * 100)
    write_log("Test end time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Write average metrics to CSV
    with open(test_csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Average', '', avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_mae, avg_srcnn_rmse, 
                         avg_bicubic_psnr, avg_bicubic_ssim, avg_bicubic_mae, avg_bicubic_rmse,
                         '', '', ''])
    
    print("\nTest results:")
    print("-" * 40)
    print(f"SRCNN average metrics:")
    print(f"PSNR: {avg_srcnn_psnr:.4f}")
    print(f"SSIM: {avg_srcnn_ssim:.4f}")
    print(f"MAE: {avg_srcnn_mae:.6f}")
    print(f"RMSE: {avg_srcnn_rmse:.6f}")
    print("-" * 40)
    print(f"Bicubic average metrics:")
    print(f"PSNR: {avg_bicubic_psnr:.4f}")
    print(f"SSIM: {avg_bicubic_ssim:.4f}")
    print(f"MAE: {avg_bicubic_mae:.6f}")
    print(f"RMSE: {avg_bicubic_rmse:.6f}")
    print("-" * 40)
    print(f"Super-resolution TIF results saved to: {tif_dir}")
    
    return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_mae, avg_srcnn_rmse, avg_bicubic_psnr, avg_bicubic_ssim, avg_bicubic_mae, avg_bicubic_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SRCNN model')
    parser.add_argument('--model_path', type=str, default=r'C:\Users\qinglin wu\Desktop\other_sr_model_train\SRCNN\model_weights\srcnn_best.pth', help='Model path')
    parser.add_argument('--test_dir', type=str, default=r'C:\Users\qinglin wu\Desktop\other_sr_model_train\datasets\test_datasets\datasets_hr\datasets_hr', help='Test data directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--bicubic_compare', action='store_true', default=True, help='Compare with bicubic interpolation results')
    parser.add_argument('--save_tiff', action='store_true', default=True, help='Save TIF file results')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Update global variables
    result_dir = args.output_dir
    tif_dir = os.path.join(result_dir, "tif_results")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(tif_dir, exist_ok=True)
    
    start_time = time.time()
    test_model(args.model_path, args.test_dir, args.num_samples, args.bicubic_compare, args.save_tiff)
    end_time = time.time()
    
    print(f"\nTest completed! Total time: {end_time - start_time:.2f} seconds")
