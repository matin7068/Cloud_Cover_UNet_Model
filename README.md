#  Cloud Cover Detection using U-Net on Sentinel-2 Images

This project implements a **U-Net based semantic segmentation model** in TensorFlow/Keras to detect **cloud pixels** in Sentinel-2 satellite imagery and estimate **cloud-cover percentage**.  
The model performs **pixel-wise binary classification** (cloud vs non-cloud) using the bands **B02, B03, B04, B08**.

---

##  Key Features
- U-Net segmentation model implemented from scratch
- Uses Sentinel-2 spectral bands (Blue, Green, Red, NIR)
- TensorFlow `tf.data` pipeline for efficient loading
- Computes **IoU, accuracy, precision, recall**
- Saves trained model as `.h5`
- Generates **cloud-cover percentage**
- Visualizes:
  - RGB image  
  - Ground-truth mask  
  - Predicted probability map  
  - Binary mask  

---

##  Model Architecture (U-Net)

The U-Net consists of:

- **Encoder / Contracting path**
  - Conv → ReLU → Conv → ReLU → MaxPool (×3 levels)

- **Bottleneck**

- **Decoder / Expanding path**
  - Upsample → Concatenate skip connections → Conv → Conv

- **Output Layer**
  - `1×1 Conv`
  - `Sigmoid activation`

This enables **fine-grained localization** while preserving global context using skip-connections.

---

## 📁 Dataset link: https://www.kaggle.com/datasets/hmendonca/cloud-cover-detection
## 📁 Dataset Structure
data/
├── train_features/
│ ├── <chip_id>/
│ │ ├── B02.tif
│ │ ├── B03.tif
│ │ ├── B04.tif
│ │ └── B08.tif
└── train_labels/
├── <chip_id>.tif


# Guwahati Sentinel-2 Cloud Mask Inference

This project runs **cloud-mask inference on multi-band Sentinel-2 images from the Guwahati region** using a trained U-Net segmentation model. The preprocessing pipeline is designed to **match the original model training setup**, so that results are consistent and meaningful.

The model expects **4-band Sentinel-2 patches (B02, B03, B04, B08) at 128×128 resolution, normalized to 0–1 using division by 65535**.  
It outputs a **binary cloud mask**.

---

## ✨ Features

- Works with **arbitrary-sized Sentinel-2 GeoTIFF images**
- Uses **Blue, Green, Red, NIR = (B02, B03, B04, B08)** in that order
- Normalizes input exactly as during training (`÷ 65535.0`)
- **Tiles the image into 128×128 patches**
- Runs inference patch-by-patch and stitches the mask back
- Generates:
  - **RGB visualization**
  - **Binary cloud mask (0=clear, 1=cloud)**
  - **Cloud-cover percentage**

---

## 📂 Input Image Format

Each input image must be a **4-band GeoTIFF**:

| Band Index | Sentinel-2 Band | Meaning |
|-----------:|-----------------|---------|
| 1 | B02 | Blue |
| 2 | B03 | Green |
| 3 | B04 | Red |
| 4 | B08 | NIR |

Example filename: S2_Guwahati_Apr2023_B2B3B4B8.tif


> **Band order matters — the model assumes exactly B02,B03,B04,B08.**

---

## 🧠 Model Details

The U-Net model was trained using:

- Sentinel-2 Surface Reflectance data
- Input channels: **B02, B03, B04, B08**
- Patch size: **128×128** for 20 epoch model, **64×64** for 10 epoch model
- Normalization:  pixel_value / 65535.0

- Sigmoid activation at the output
- Binary threshold: **0.5**

This repo reproduces the same pipeline during inference.

---

## 🔧 Installation

```bash
pip install numpy rasterio matplotlib tensorflow
 



