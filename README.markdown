# SRCNN DEM Super-Resolution Project

This project implements a Super-Resolution Convolutional Neural Network (SRCNN) for enhancing the resolution of Digital Elevation Model (DEM) data stored in GeoTIFF format. The model is trained to upscale low-resolution (16x16) DEM images to high-resolution (64x64) images, improving the quality of topographic data.

## Project Structure

- `train_srcnn.py`: Contains the training script for the SRCNN model, including dataset loading, model definition, training loop, and evaluation metrics.
- `test_srcnn.py`: Implements the testing pipeline to evaluate the trained SRCNN model on a test dataset, compute performance metrics (PSNR, SSIM, MAE, RMSE), and generate visualizations.
- `logs/`: Directory storing training and validation logs (`training_log.txt`, `validation_log.txt`, `test_results.txt`) and a CSV file with test metrics (`test_metrics.csv`).
- `model_weights/`: Directory for saving trained model weights (`srcnn_best.pth`, `srcnn_best_loss.pth`, and epoch-specific checkpoints).
- `results/`: Directory for storing visualization outputs (triptych comparisons) and super-resolved GeoTIFF files (`tif_results/`).

## Requirements

To run this project, ensure you have the following dependencies installed:

```bash
pip install torch torchvision numpy matplotlib scikit-image rasterio tqdm opencv-python Pillow
```

- **Python**: 3.8 or higher
- **PyTorch**: 1.9 or higher (with CUDA support for GPU training)
- **Hardware**: GPU recommended for faster training, but CPU is supported.

## Dataset Preparation

The dataset should consist of GeoTIFF files containing DEM data with the following specifications:

- **Training Data**: High-resolution DEM images (64x64 pixels) stored in a directory (e.g., `datasets/train_datasets/datasets_hr/`).
- **Test Data**: High-resolution DEM images (64x64 pixels) stored in a separate directory (e.g., `datasets/test_datasets/datasets_hr/`).

Each GeoTIFF file should represent a single DEM patch. The dataset class (`DEMDataset`) normalizes the data and generates low-resolution versions (16x16) using nearest-neighbor downsampling during training and testing.

## Training the Model

To train the SRCNN model, run the `train_srcnn.py` script with appropriate arguments:

```bash
python train_srcnn.py --train_dir /path/to/train/data --test_dir /path/to/test/data
```

### Training Parameters

- `--train_dir`: Path to the training dataset directory.
- `--test_dir`: Path to the test dataset directory.
- `--batch_size`: Batch size for training (default: 16).
- `--epochs`: Number of training epochs (default: 150).
- `--lr`: Learning rate (default: 1e-4).
- `--output_dir`: Directory to save model weights and logs (default: `model_weights/`).

### Training Process

- The model uses a three-layer CNN architecture (9x9, 1x1, 5x5 convolutions) to learn the mapping from low-resolution to high-resolution DEM images.
- Loss function: Mean Squared Error (MSE).
- Optimizer: SGD with momentum (0.9) and weight decay (1e-4).
- Learning rate scheduling: Reduces learning rate by half if the loss plateaus for 5 epochs.
- Gradient clipping: Applied with a maximum norm of 1.0 to stabilize training.
- Metrics: PSNR, SSIM, MAE, and RMSE are computed for both training and validation sets after each epoch.
- Checkpoints: Model weights are saved every 10 epochs, with the best models (based on validation PSNR and training loss) saved as `srcnn_best.pth` and `srcnn_best_loss.pth`.

### Outputs

- **Logs**: Training and validation metrics are logged in `logs/training_log.txt` and `logs/validation_log.txt`.
- **Model Weights**: Saved in `model_weights/` with filenames indicating epoch number or best performance.

## Testing the Model

To test the trained model, run the `test_srcnn.py` script:

```bash
python test_srcnn.py --model_path /path/to/srcnn_best.pth --test_dir /path/to/test/data
```

### Testing Parameters

- `--model_path`: Path to the trained model weights (e.g., `model_weights/srcnn_best.pth`).
- `--test_dir`: Path to the test dataset directory.
- `--num_samples`: Number of samples to visualize (default: 5).
- `--bicubic_compare`: Compare SRCNN results with bicubic interpolation (default: True).
- `--save_tiff`: Save super-resolved images as GeoTIFF files (default: True).
- `--output_dir`: Directory to save results (default: `results/`).

### Testing Process

- The script evaluates the model on the test dataset, computing PSNR, SSIM, MAE, and RMSE for both SRCNN and bicubic interpolation.
- Visualizations: Triptych images (low-resolution, super-resolved, ground truth) are generated for a random subset of samples, saved in `results/`.
- GeoTIFF Output: Super-resolved DEMs are saved as GeoTIFF files in `results/tif_results/`, preserving geospatial metadata if available.
- Metrics are logged in `logs/test_results.txt` and saved as a CSV in `logs/test_metrics.csv`.

## Evaluation Metrics

The following metrics are used to evaluate the model:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of the reconstructed image (higher is better).
- **SSIM (Structural Similarity Index)**: Assesses the structural similarity between images (range: 0 to 1, higher is better).
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between predicted and ground truth values (lower is better).
- **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared differences (lower is better).

## Visualizations

The testing script generates triptych visualizations for selected samples, showing:

- **Low-Resolution Image** (16x16, no upsampling).
- **Super-Resolved Image** (64x64, from SRCNN).
- **Ground Truth** (64x64, high-resolution reference).

These visualizations use the `terrain` colormap with a shared colorbar, saved as PNG files in the `results/` directory.

## Usage Notes

- Ensure the input GeoTIFF files are 64x64 pixels for consistency with the model architecture.
- The dataset class assumes single-band GeoTIFF files. Multi-band files may require preprocessing.
- For large datasets, adjust the `batch_size` and `num_workers` to optimize memory usage and training speed.
- If running on a CPU, training and testing may be significantly slower. Consider reducing the batch size or using a smaller dataset.

## Example Commands

```bash
# Train the model
python train_srcnn.py --train_dir datasets/train_datasets/datasets_hr --test_dir datasets/test_datasets/datasets_hr --epochs 150 --batch_size 16

# Test the model
python test_srcnn.py --model_path model_weights/srcnn_best.pth --test_dir datasets/test_datasets/datasets_hr --num_samples 5 --bicubic_compare --save_tiff
```

## Future Improvements

- Add support for multi-band GeoTIFF files.
- Implement additional super-resolution architectures (e.g., SRResNet, EDSR).
- Optimize the model for larger DEM patches or variable input sizes.
- Integrate data augmentation techniques to improve model robustness.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.