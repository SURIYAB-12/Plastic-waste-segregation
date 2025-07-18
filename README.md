# ♻️ Plastic Type Classification using MobileNetV2

A deep learning model that classifies plastic waste into 7 categories using **Transfer Learning** with **MobileNetV2**. This project helps automate plastic identification to support recycling and sustainability.

---

## 🚀 Overview

🧠 **Model:** MobileNetV2 (pre-trained on ImageNet)  
🧾 **Classes:** `hdpe`, `ldpe`, `other`, `pet`, `pp`, `ps`, `pvc`  
🖼️ **Input Size:** 224 x 224 x 3  
📦 **Output:** Trained model file: `my_trained_model.h5`  

---
## 📂 Project Structure

plastic-type-classifier/
│ └── TRAINING/ # Subfolders for each plastic class
│ ├── hdpe/
│ ├── ldpe/
│ ├── other/
│ ├── pet/
│ ├── pp/
│ ├── ps/
│ └── pvc/
│
├── train_model.py # Main training script (data prep + training)
├── my_trained_model.h5 # Final trained model (MobileNetV2-based)
├── README.md # Project documentation (you’re reading this!)


Each folder name is treated as a label for image classification.

---

## 🧠 Model Architecture

MobileNetV2 (frozen)
│
├── Flatten()
│
├── Dense(128, activation='relu')
│
└── Dense(7, activation='softmax')


🔒 Base model layers are frozen

🔍 Custom classifier head is trained on your dataset

🧪 Dataset Summary
Split	Count	Shape
Training	72	(72, 224, 224, 3)
Validation	16	(16, 224, 224, 3)
Testing	16	(16, 224, 224, 3)
Classes	7	See below

Class Names:
['hdpe', 'ldpe', 'other', 'pet', 'pp', 'ps', 'pvc']

🛠️ Setup & Usage
📦 1. Install Dependencies
  
pip install tensorflow numpy pillow scikit-learn

🗂️ 2. Clone Repository

git clone https://github.com/your-username/plastic-type-classifier.git
cd plastic-type-classifier

🧑‍🏫 3. Train the Model

python train_model.py

📁 This script will:

Preprocess data

Train MobileNetV2 for 10 epochs

Save the model to my_trained_model.h5

## 📈 Training Performance

During training, the model achieved **100% accuracy on the training set** by the 5th epoch. However, **validation accuracy plateaued around 37–44%**, indicating potential overfitting.

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------------|-----------------|
| 1     | 22.22%            | 25.00%              | 15.58           |
| 3     | 87.50%            | 43.75%              | 6.35            |
| 5     | 100.00%           | 43.75%              | 6.98            |
| 10    | 100.00%           | 37.50%              | 7.07            |

> ⚠️ **Note**: The training results suggest that while the model learns the training data well, it does not generalize effectively to unseen data. This is common in small datasets and can be addressed through the following enhancements:

### 📌 Next Steps to Improve Generalization

- 🧪 **Data Augmentation** (rotation, flipping, zooming, etc.)
- 🧱 **Regularization Techniques**: Dropout, L2 regularization
- 🧠 **Fine-tuning** the deeper layers of MobileNetV2




🔗 Complete Project Files
📁 To access the complete project files including datasets, models, and resources, follow the link below:

👉 Click here to open the full project folder on Google Drive
- 📊 **Class balancing** if some categories have fewer samples
- 🖼️ **Increasing dataset size** with more labeled plastic images

> These steps will help the model learn more generalizable patterns and improve real-world performance.




## 📎 Complete Project Access

[🔗 **Click here to access the full project folder on Google Drive**](https://drive.google.com/drive/folders/1LMweXUgtUzd-HhUUUxyuC-hadok8pZ5-?usp=sharing)

> This folder contains:
> - Training images
> - The final `.h5` model
> - Related resources
