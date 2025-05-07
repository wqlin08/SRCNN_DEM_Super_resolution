import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
from rasterio import open as rasterio_open
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
model_dir = "model_weights"
os.makedirs(model_dir, exist_ok=True)


log_file = os.path.join(log_dir, "training_log.txt")
val_log_file = os.path.join(log_dir, "validation_log.txt") 


BATCH_SIZE = 16
EPOCHS = 150
LR = 1e-4  
HR_SIZE = 64  
LR_SIZE = 16  

# SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 9×9, 1×64, ReLU
        # 1×1, 64×32, ReLU
        # 5×5, 32×1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# datasets
class DEMDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        self.file_list = []
        for file in os.listdir(data_dir):
            if file.endswith('.tif'):
                self.file_list.append(os.path.join(data_dir, file))
                
        print(f"find {len(self.file_list)} tif in {data_dir}")
        self.value_ranges = {} 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with rasterio_open(file_path) as src:
            hr_img = src.read(1).astype(np.float32)
            
        if hr_img.shape != (HR_SIZE, HR_SIZE):
            raise ValueError(f"hope the images size are {HR_SIZE}x{HR_SIZE}, but {hr_img.shape}")
            
        min_val = hr_img.min()
        max_val = hr_img.max()
        self.value_ranges[file_path] = (min_val, max_val)
            
        hr_img_normalized = (hr_img - min_val) / (max_val - min_val) if max_val > min_val else hr_img
        
        # useing nearnest down scalling
        hr_tensor = torch.from_numpy(hr_img_normalized).unsqueeze(0)
        lr_tensor = nn.functional.interpolate(
            hr_tensor.unsqueeze(0), 
            size=(LR_SIZE, LR_SIZE),
            mode='nearest'
        ).squeeze(0) 
        
        range_info = torch.tensor([min_val, max_val], dtype=torch.float32)
        
        return lr_tensor, hr_tensor, range_info

def evaluate_model(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_mae = 0
    total_rmse = 0
    n_samples = 0
    n_valid_samples = 0 
    
    with torch.no_grad():
        for lr_imgs, hr_imgs, range_info in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            lr_upscale = nn.functional.interpolate(lr_imgs, size=(HR_SIZE, HR_SIZE), mode='bicubic', align_corners=False)
            sr_imgs = model(lr_upscale)
            
            sr_np = sr_imgs.cpu().numpy()
            hr_np = hr_imgs.cpu().numpy()
            
            for i in range(sr_np.shape[0]):
                sr_img = np.squeeze(sr_np[i])
                hr_img = np.squeeze(hr_np[i])
                
                sr_img = np.clip(sr_img, 0, 1)
                hr_img = np.clip(hr_img, 0, 1)
                
                min_val, max_val = range_info[i].numpy()
                value_range = max_val - min_val
                
                if value_range < 1e-6:
                    value_range = 1.0
                
                sr_img_original = sr_img * value_range + min_val
                hr_img_original = hr_img * value_range + min_val
                
                if np.array_equal(sr_img_original, hr_img_original):
                    sr_img_original = sr_img_original + np.random.normal(0, 1e-6, sr_img_original.shape)
                
                mse = mean_squared_error(hr_img_original, sr_img_original)
                if mse == 0:
                    mse = 1e-10
                
                try:
                    current_psnr = psnr(hr_img_original, sr_img_original, data_range=value_range)
                    
                    if np.isinf(current_psnr) or np.isnan(current_psnr):
                        current_psnr = 20 * np.log10(value_range / np.sqrt(mse))
                    
                    if np.isinf(current_psnr) or np.isnan(current_psnr):
                        current_psnr = 0
                    else:
                        total_psnr += current_psnr
                        n_valid_samples += 1
                        
                    current_ssim = ssim(hr_img_original, sr_img_original, data_range=value_range)

                    if np.isnan(current_ssim):
                        current_ssim = 0
                    else:
                        total_ssim += current_ssim
                except Exception as e:
                    print(f"Calculation error: {e}")
                    current_psnr = 0
                    current_ssim = 0
                
                current_mae = mean_absolute_error(hr_img_original, sr_img_original)
                total_mae += current_mae
                
                current_rmse = np.sqrt(mse)
                total_rmse += current_rmse
                
                n_samples += 1
    
    if n_valid_samples == 0:
        n_valid_samples = 1
        
    avg_psnr = total_psnr / n_valid_samples
    avg_ssim = total_ssim / n_valid_samples
    avg_mae = total_mae / n_samples
    avg_rmse = total_rmse / n_samples
    
    return avg_psnr, avg_ssim, avg_mae, avg_rmse

def write_log(message, is_validation=False):
    log_f = val_log_file if is_validation else log_file
    with open(log_f, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"devide used: {device}")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Train start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':^6}{'Loss':^15}{'PSNR':^10}{'SSIM':^10}{'MAE':^15}{'RMSE':^15}\n")
        f.write("-" * 80 + "\n")

    with open(val_log_file, 'w', encoding='utf-8') as f:
        f.write("Verification start time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':^6}{'PSNR':^10}{'SSIM':^10}{'MAE':^15}{'RMSE':^15}\n")
        f.write("-" * 80 + "\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # data dir
    train_data_dir = ""
    test_data_dir = ""
    
    train_dataset = DEMDataset(train_data_dir, is_train=True)
    test_dataset = DEMDataset(test_data_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(test_dataset)}")
    
    model = SRCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created SRCNN model with {total_params:,} parameters")
    print(str(model))
    
    write_log("model:")
    write_log(str(model))
    write_log("-" * 80)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    print("Training...")
    best_psnr = 0
    best_loss = float('inf') 
    
    max_grad_norm = 1.0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        batch_losses = [] 
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for lr_imgs, hr_imgs, _ in train_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                lr_upscale = nn.functional.interpolate(lr_imgs, size=(HR_SIZE, HR_SIZE), mode='bicubic', align_corners=False)

                optimizer.zero_grad()
                sr_imgs = model(lr_upscale)
                
                loss = criterion(sr_imgs, hr_imgs)
                batch_losses.append(loss.item())
                epoch_loss += loss.item()
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

        train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        train_psnr, train_ssim, train_mae, train_rmse = evaluate_model(model, train_loader_eval, device)
        print(f"Train PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")

        val_psnr, val_ssim, val_mae, val_rmse = evaluate_model(model, test_loader, device)
        print(f"Validation PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        
        scheduler.step(avg_loss)
        
        train_log_message = f"{epoch+1:^6}{avg_loss:^15.6f}{train_psnr:^10.4f}{train_ssim:^10.4f}{train_mae:^15.6f}{train_rmse:^15.6f}"
        write_log(train_log_message)

        val_log_message = f"{epoch+1:^6}{val_psnr:^10.4f}{val_ssim:^10.4f}{val_mae:^15.6f}{val_rmse:^15.6f}"
        write_log(val_log_message, is_validation=True)

        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(model_dir, f"srrcnn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"model save {model_save_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_loss_path = os.path.join(model_dir, "srcnn_best_loss.pth")
            torch.save(model.state_dict(), best_model_loss_path)
            print(f"best loss model save, Loss: {best_loss:.6f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_model_path = os.path.join(model_dir, "srcnn_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"best psnr model save, PSNR: {best_psnr:.4f}")
    
    write_log("-" * 80)
    write_log("Train end time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    write_log("-" * 80, is_validation=True)
    write_log("Val end time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_validation=True)
    print("Training Completed！")

if __name__ == "__main__":
    main() 
