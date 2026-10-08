<div align="center">

# 🧠 CIFAR-10 Image Classification using Deep Learning

### CNN from Scratch • Data Augmentation • Transfer Learning • Fine-Tuning

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=36BCF7&center=true&vCenter=true&width=850&lines=Deep+Learning+PR4+%7C+Project+A;CNN+from+Scratch+on+CIFAR-10;Data+Augmentation+%2B+Transfer+Learning;MobileNetV2+Feature+Extraction+%26+Fine-Tuning;Best+Test+Accuracy%3A+83.73%25" alt="Typing SVG" />

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![CIFAR10](https://img.shields.io/badge/Dataset-CIFAR--10-20BEFF?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

<br>

> **A complete deep-learning image-classification project progressing from fundamental CNN operations to data augmentation, transfer learning, fine-tuning, and residual connections.**

</div>

---

# 📌 Project Overview

This project implements a complete **image classification pipeline on the CIFAR-10 dataset** using **TensorFlow and Keras**.

The project begins with the fundamentals of Convolutional Neural Networks and gradually progresses toward more advanced deep-learning techniques.

The implementation covers:

- 📊 CIFAR-10 exploration and preprocessing
- 🧠 CNN vs ANN concepts
- 🔍 Manual convolution
- 🖼️ Padding and strides
- 🗺️ Feature-map visualization
- 📉 MaxPooling and AveragePooling
- 🏛️ LeNet-5 architecture
- 🚀 Custom CNN from scratch
- 🔄 Backpropagation concepts
- 📈 CNN training and evaluation
- 🎨 Data augmentation
- 🧩 Keras augmentation layers
- ⚡ MobileNetV2 transfer learning
- 🔓 Fine-tuning
- 🔗 Residual / skip connections
- 📊 Six-model performance comparison

The primary objective is not only to obtain high classification accuracy but also to understand **how CNN architectures evolve from basic convolution operations to modern transfer-learning approaches**.

---

# 🏆 Final Result

<div align="center">

## 🥇 Best Performing Model

### CNN Scratch + ImageDataGenerator

| Metric | Result |
|:---:|:---:|
| 🏆 **Test Accuracy** | **83.73%** |
| 🧠 Architecture | Custom CNN |
| 🎨 Augmentation | ImageDataGenerator |
| 🔢 Parameters | 175,018 |
| 🔄 Transfer Learning | No |

### ⭐ **83.73% Test Accuracy**

</div>

The custom CNN trained with **ImageDataGenerator augmentation** achieved the highest test accuracy among all six evaluated approaches.

---

# 📂 Dataset — CIFAR-10

The project uses the built-in **CIFAR-10 dataset** available through TensorFlow/Keras.

```python
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

### Dataset Information

| Property | Value |
|---|---:|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Image Resolution | 32 × 32 |
| Channels | 3 — RGB |
| Number of Classes | 10 |
| Original Pixel Range | 0–255 |
| Normalized Range | 0–1 |

---

# 🏷️ CIFAR-10 Classes

The dataset contains the following ten classes:

```text
0 → Airplane
1 → Automobile
2 → Bird
3 → Cat
4 → Deer
5 → Dog
6 → Frog
7 → Horse
8 → Ship
9 → Truck
```

CIFAR-10 is challenging because every image has a resolution of only **32 × 32 pixels**.

Visually related classes such as:

```text
Cat        ↔ Dog
Automobile ↔ Truck
```

can therefore be difficult for a classifier to distinguish.

---

# 🧭 Complete Project Workflow

```text
                 CIFAR-10 Dataset
                        │
                        ▼
              Data Exploration
                        │
                        ▼
                Preprocessing
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
         Normalize           One-Hot Encode
              │                   │
              └─────────┬─────────┘
                        ▼
              CNN Fundamentals
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    Convolution      Padding        Strides
         │
         ▼
    Feature Maps
         │
         ▼
       Pooling
         │
         ▼
       LeNet-5
         │
         ▼
     Custom CNN
         │
         ▼
       Training
         │
         ▼
      Evaluation
         │
    ┌────┴─────────────┐
    ▼                  ▼
Augmentation     Transfer Learning
    │                  │
    ▼                  ▼
ImageDataGen      MobileNetV2
    │                  │
    ▼             Feature Extraction
Keras Layers           │
                       ▼
                   Fine-Tuning
                       │
                       ▼
                Model Comparison
                       │
                       ▼
                 Final Results
```

---

# 1️⃣ Data Loading, EDA & Preprocessing

The first stage focuses on understanding and preparing CIFAR-10.

### Operations Performed

- Loaded CIFAR-10
- Inspected dataset shapes
- Checked data types
- Examined class labels
- Displayed samples from all 10 classes
- Visualized pixel distributions
- Normalized images
- One-hot encoded labels

### Normalization

Original pixels:

```text
0 → 255
```

Normalized pixels:

```text
0 → 1
```

Implemented as:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

Normalization makes optimization more stable and prevents large input values from dominating gradient calculations.

---

# 🧠 CNN vs ANN for Image Classification

Traditional ANNs generally require images to be flattened.

```text
32 × 32 × 3
      │
      ▼
    Flatten
      │
      ▼
3072-dimensional vector
```

Flattening removes explicit spatial relationships between neighboring pixels.

CNNs instead apply small learnable kernels across local regions:

```text
Image
  │
  ▼
3×3 Kernel
  │
  ▼
Local Feature Detection
  │
  ▼
Feature Map
```

CNNs therefore provide major advantages for image data:

- Local connectivity
- Parameter sharing
- Spatial feature extraction
- Hierarchical representation learning
- Better handling of image patterns

---

# 2️⃣ Convolution, Padding & Strides

## 🔍 Manual Convolution

A manual convolution experiment was implemented using a **3 × 3 edge-detection kernel**.

```python
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
```

The kernel slides across the image and computes a local dot product at every valid location.

This demonstrates the fundamental operation performed internally by `Conv2D`.

---

## 🖼️ Padding

Two padding strategies were investigated.

### VALID Padding

```text
Input  : 32 × 32
Kernel : 3 × 3

Output : 30 × 30
```

### SAME Padding

```text
Input  : 32 × 32
Kernel : 3 × 3

Output : 32 × 32
```

`SAME` padding is useful when stacking multiple convolution layers because it prevents spatial dimensions from shrinking too quickly.

---

## 👣 Strides

Stride determines how far a kernel moves after each convolution operation.

```text
Stride = 1
32 × 32 → 32 × 32

Stride = 2
32 × 32 → 16 × 16
```

A larger stride can therefore perform spatial downsampling.

---

# 3️⃣ Feature Map Visualization

A convolutional layer containing **16 filters** was applied to a CIFAR-10 image.

```python
Conv2D(
    16,
    (3, 3),
    activation="relu",
    padding="same"
)
```

Output:

```text
Input  → (1, 32, 32, 3)
Output → (1, 32, 32, 16)
```

Each filter creates an individual feature map.

Early CNN layers can learn features such as:

```text
Edges
  ↓
Gradients
  ↓
Textures
  ↓
Shapes
  ↓
Higher-Level Patterns
```

### 🖼️ Feature Maps

![Feature Maps](plots/feature_maps.png)

---

# 4️⃣ Pooling Operations

Two pooling strategies were studied.

## MaxPooling

MaxPooling keeps the strongest activation from each pooling window.

```text
32 × 32
   │
   ▼
MaxPooling(2×2)
   │
   ▼
16 × 16
```

It is useful for preserving prominent features while reducing spatial dimensions.

## AveragePooling

AveragePooling calculates the average activation within each local pooling region.

It produces smoother feature representations compared with MaxPooling.

---

# 🔄 Flatten Operation

After convolution and pooling:

```text
16 × 16 × 16
      │
      ▼
    Flatten
      │
      ▼
     4096
```

Flatten converts multidimensional feature maps into a one-dimensional vector that can be passed into Dense layers.

---

# 5️⃣ LeNet-5 Architecture

A CIFAR-10 adaptation of the historical **LeNet-5 CNN architecture** was implemented.

```text
Input (32×32×3)
       │
       ▼
Conv2D — 6 Filters — 5×5 — Tanh
       │
       ▼
AveragePooling2D
       │
       ▼
Conv2D — 16 Filters — 5×5 — Tanh
       │
       ▼
AveragePooling2D
       │
       ▼
Flatten
       │
       ▼
Dense — 120 — Tanh
       │
       ▼
Dense — 84 — Tanh
       │
       ▼
Dense — 10 — Softmax
```

### Parameter Verification

First convolution:

```text
(5 × 5 × 3 + 1) × 6
= 456 parameters
```

Second convolution:

```text
(5 × 5 × 6 + 1) × 16
= 2,416 parameters
```

### Total Parameters

```text
62,006
```

### Final Test Accuracy

```text
53.52%
```

---

# 6️⃣ Custom CNN Architecture

A more modern CNN was developed for CIFAR-10.

```text
Input — 32×32×3
        │
        ▼
Conv2D(32, 3×3, ReLU)
        │
        ▼
Conv2D(32, 3×3, ReLU)
        │
        ▼
MaxPooling2D
        │
        ▼
Conv2D(64, 3×3, ReLU)
        │
        ▼
Conv2D(64, 3×3, ReLU)
        │
        ▼
MaxPooling2D
        │
        ▼
Conv2D(128, 3×3, ReLU)
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dense(256, ReLU)
        │
        ▼
Dense(10, Softmax)
```

### Parameters

```text
175,018
```

Using `GlobalAveragePooling2D` instead of a large Flatten-based classification head helps keep the architecture compact.

---

# 7️⃣ CNN Training

The custom CNN was compiled using:

```python
model_cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

Training configuration:

```text
Maximum Epochs     : 100
Batch Size         : 64
Validation Split   : 10%
Optimizer          : Adam
Loss               : Categorical Crossentropy
```

---

# ⏹️ EarlyStopping

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

EarlyStopping prevents unnecessary training when validation performance stops improving.

---

# 📉 ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```

The learning rate is gradually reduced when validation loss reaches a plateau, allowing smaller optimization steps near convergence.

---

# 📈 Training Performance

![CNN Training Curves](plots/training_curves.png)

The training curves help identify:

- Convergence
- Overfitting
- Validation performance
- Effect of learning-rate reduction
- Early-stopping behavior

---

# 8️⃣ Model Evaluation

The trained CNN was evaluated using the **10,000 unseen CIFAR-10 test images**.

Evaluation included:

- Test loss
- Test accuracy
- Confusion matrix
- Most-confused class pairs
- Misclassified-image analysis

---

# 🧩 Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

The confusion matrix provides class-level insight into model performance.

Visually similar categories can be more challenging because CIFAR-10 images contain only **32 × 32 pixels**.

---

# ❌ Misclassified Images

![Misclassified Images](plots/misclassified_images.png)

Misclassified samples are useful for understanding the model's weaknesses.

Common causes include:

- Low image resolution
- Similar object shapes
- Similar textures
- Complex backgrounds
- Partial objects
- Ambiguous visual features

---

# 9️⃣ Data Augmentation

Data augmentation was introduced to reduce overfitting and improve generalization.

The `ImageDataGenerator` configuration used:

```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest"
)
```

These transformations create different versions of the original training images without changing their labels.

---

# 🎨 Augmentation Visualization

![Data Augmentation](plots/augmentation_grid.png)

The augmentation pipeline includes:

```text
Original Image
      │
      ├── Rotation
      ├── Width Shift
      ├── Height Shift
      ├── Horizontal Flip
      └── Zoom
```

This effectively increases the diversity of the training data.

---

# 🔥 Effect of ImageDataGenerator

The custom CNN without augmentation achieved:

```text
78.84%
```

After training the same architecture using `ImageDataGenerator`:

```text
83.73%
```

### Improvement

```text
83.73 - 78.84 = +4.89 percentage points
```

This was the **highest test accuracy obtained in the project**.

---

# 🔟 Modern Keras Augmentation Layers

A second augmentation strategy was implemented directly inside the model.

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])
```

Advantages of model-integrated augmentation include:

- Runs as part of the TensorFlow pipeline
- Automatically active during training
- Automatically inactive during inference
- Easy model integration

### Result

```text
Test Accuracy = 76.20%
```

In this experiment, the `ImageDataGenerator` approach performed better.

---

# 1️⃣1️⃣ Transfer Learning

The project next explored **MobileNetV2**, using weights learned from ImageNet.

Transfer learning allows knowledge from a large pre-trained model to be reused for a new classification task.

Two strategies were evaluated:

```text
MobileNetV2
    │
    ├── Feature Extraction
    │
    └── Fine-Tuning
```

---

# ⚡ MobileNetV2 Feature Extraction

The MobileNetV2 convolutional base was initially frozen.

```python
base_model.trainable = False
```

The classification head consisted of:

```text
MobileNetV2
     │
     ▼
GlobalAveragePooling2D
     │
     ▼
Dense(128, ReLU)
     │
     ▼
Dense(10, Softmax)
```

### Result

```text
Test Accuracy = 79.13%
Parameters    = 2,423,242
```

---

# 🔓 MobileNetV2 Fine-Tuning

After feature extraction, the upper MobileNetV2 layers were unfrozen.

```text
Top 30 MobileNetV2 Layers
           │
           ▼
       Unfrozen
```

Fine-tuning used a very small learning rate:

```python
Adam(learning_rate=1e-5)
```

This protects the useful pre-trained features from being changed too aggressively.

### Result

```text
Test Accuracy = 81.24%
Parameters    = 2,423,242
```

### Improvement over Feature Extraction

```text
81.24 - 79.13 = +2.11 percentage points
```

---

# 📈 Transfer Learning Curves

![Transfer Learning Curves](plots/transfer_learning_curves.png)

The graph combines:

```text
Feature Extraction
       │
       ▼
Fine-Tuning Start
       │
       ▼
Fine-Tuning
```

This makes it possible to observe how model performance changes after the pre-trained layers become trainable.

---

# 1️⃣2️⃣ Residual Connection

A minimal residual block was implemented using the Keras Functional API.

```text
                         ┌── Conv2D(1×1) ──────────────┐
                         │                              │
Input ───────────────────┤                              ├── Add → ReLU
                         │                              │
                         └── Conv3×3 → ReLU → Conv3×3 ─┘
```

Residual connections allow information and gradients to travel through shortcut paths.

They are especially useful in deep networks because they help reduce optimization problems associated with vanishing gradients.

Residual connections are a central concept behind architectures such as **ResNet**.

---

# 📊 Final Model Comparison

| # | Model | Architecture | Augmentation | Transfer Learning | Test Accuracy | Parameters |
|---:|---|---|---|---|---:|---:|
| 1 | LeNet-5 | LeNet-5 | No | No | **53.52%** | 62,006 |
| 2 | CNN Scratch No Aug | Custom CNN | No | No | **78.84%** | 175,018 |
| 🥇 3 | **CNN Scratch + Aug** | **Custom CNN** | **ImageDataGenerator** | **No** | **83.73% 🏆** | **175,018** |
| 4 | CNN + Keras Aug Layers | Custom CNN | Keras Layers | No | **76.20%** | 175,018 |
| 5 | MobileNetV2 Feature Extraction | MobileNetV2 | No | Feature Extraction | **79.13%** | 2,423,242 |
| 6 | MobileNetV2 Fine-Tuned | MobileNetV2 | No | Fine-Tuning | **81.24%** | 2,423,242 |

---

# 🏆 Accuracy Ranking

```text
🥇 CNN Scratch + ImageDataGenerator     83.73% ██████████████████████████████████████████
🥈 MobileNetV2 Fine-Tuned               81.24% █████████████████████████████████████████
🥉 MobileNetV2 Feature Extraction       79.13% ████████████████████████████████████████
   CNN Scratch No Aug                   78.84% ███████████████████████████████████████
   CNN + Keras Aug Layers               76.20% ██████████████████████████████████████
   LeNet-5                              53.52% ███████████████████████████
```

---

# 🔬 Key Experimental Findings

### 🥇 Best Overall Accuracy

**CNN Scratch + ImageDataGenerator — 83.73%**

This model achieved the highest test accuracy in the experiment.

### 📈 Benefit of Data Augmentation

```text
Without Augmentation : 78.84%
With ImageDataGen    : 83.73%
Improvement          : +4.89 percentage points
```

### ⚡ Benefit of Fine-Tuning

```text
Feature Extraction : 79.13%
Fine-Tuning        : 81.24%
Improvement        : +2.11 percentage points
```

### 🪶 Most Parameter-Efficient Model

```text
LeNet-5
Parameters: 62,006
Accuracy:   53.52%
```

It is the smallest evaluated network, although its accuracy was substantially lower.

### 💡 Strong Accuracy-to-Size Result

The augmented custom CNN achieved:

```text
Accuracy   : 83.73%
Parameters : 175,018
```

while the MobileNetV2 models contained:

```text
Parameters : 2,423,242
```

In this experiment, the much smaller augmented custom CNN produced the highest test accuracy.

---

# 🛠️ Technologies Used

<div align="center">

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🧠 TensorFlow | Deep Learning |
| 🔥 Keras | CNN Model Development |
| 🔢 NumPy | Numerical Operations |
| 📊 Pandas | Results & Comparison |
| 📈 Matplotlib | Visualization |
| 🧮 SciPy | Manual Convolution |
| 🤖 Scikit-learn | Confusion Matrix |
| 📓 Jupyter Notebook | Development Environment |

</div>

---

# 📦 Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project directory:

```bash
cd DL_PR4A_CIFAR10
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
DL_PR4A.ipynb
```

Then select:

```text
Kernel → Restart Kernel and Run All Cells
```

---

# 📋 requirements.txt

A typical environment for this project requires:

```text
tensorflow
numpy
pandas
matplotlib
scipy
scikit-learn
jupyter
```

---

# ▶️ How to Run

```bash
# 1. Clone repository
git clone YOUR_REPOSITORY_URL

# 2. Open project
cd DL_PR4A_CIFAR10

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook

# 5. Open
DL_PR4A.ipynb
```

The CIFAR-10 dataset is loaded automatically through TensorFlow/Keras, so no separate dataset download is required.

---

# 🎥 Project Demo Video

<div align="center">

### ▶️ Watch Complete Project Explanation

[![Project Demo](https://img.shields.io/badge/▶_WATCH_PROJECT_DEMO-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](YOUR_VIDEO_LINK)

</div>

> Replace `YOUR_VIDEO_LINK` with your Google Drive, YouTube, or other project-demo video link.

---

# 📸 Project Visualizations

The repository contains important visual outputs generated during the experiment.

| Visualization | Purpose |
|---|---|
| 🗺️ Feature Maps | Understand convolution filters |
| 🎨 Augmentation Grid | View transformed training samples |
| 📈 Training Curves | Analyze training and validation performance |
| 🧩 Confusion Matrix | Analyze class-level predictions |
| ❌ Misclassified Images | Understand model errors |
| ⚡ Transfer Learning Curves | Compare feature extraction and fine-tuning |
| 📊 Results Table | Compare all models |

---

# 🧠 Concepts Covered

```text
Deep Learning
│
├── Artificial Neural Networks
│
├── Convolutional Neural Networks
│   ├── Convolution
│   ├── Kernels
│   ├── Feature Maps
│   ├── Padding
│   ├── Strides
│   └── Pooling
│
├── CNN Architectures
│   ├── LeNet-5
│   ├── Custom CNN
│   ├── MobileNetV2
│   └── Residual Connections
│
├── Training
│   ├── Backpropagation
│   ├── Adam
│   ├── Categorical Crossentropy
│   ├── EarlyStopping
│   └── ReduceLROnPlateau
│
├── Regularization
│   └── Data Augmentation
│
└── Transfer Learning
    ├── Feature Extraction
    └── Fine-Tuning
```

---

# 🎯 Project Highlights

<div align="center">

| 🚀 Highlight | 📊 Result |
|---|---|
| Dataset | CIFAR-10 |
| Classes | 10 |
| Models Compared | 6 |
| Best Model | CNN Scratch + Aug |
| Best Accuracy | **83.73%** |
| Best CNN Parameters | **175,018** |
| Transfer Model | MobileNetV2 |
| Fine-Tuned Accuracy | **81.24%** |
| Framework | TensorFlow / Keras |

</div>

---

# 🔮 Future Improvements

Potential extensions of this project include:

- Batch Normalization
- Dropout regularization
- Deeper residual networks
- Learning-rate scheduling
- More advanced augmentation
- Hyperparameter optimization
- EfficientNet transfer learning
- ResNet transfer learning
- Model ensembling
- Grad-CAM visualization
- Web-based inference application

---

# 🎓 Learning Outcomes

After completing this project, the following concepts were implemented practically:

- Loading and preprocessing image datasets
- Understanding convolution mathematically
- Applying padding and strides
- Visualizing CNN feature maps
- Understanding pooling operations
- Building LeNet-5
- Designing a CNN from scratch
- Understanding CNN backpropagation
- Training with callbacks
- Evaluating image classifiers
- Analyzing confusion matrices
- Applying data augmentation
- Using modern Keras augmentation layers
- Implementing transfer learning
- Performing fine-tuning
- Comparing deep-learning models
- Creating residual connections with the Functional API

---

# 📌 Conclusion

This project demonstrates a complete progression through **Convolutional Neural Networks for image classification**.

The experiment started with CNN fundamentals such as convolution, padding,
strides, pooling and feature maps. It then progressed through LeNet-5 and a
custom CNN architecture before exploring data augmentation and MobileNetV2
transfer learning.

Among the six evaluated approaches, the best result was:

<div align="center">

# 🏆 83.73%

### CNN Scratch + ImageDataGenerator

**175,018 Parameters**

</div>

The custom augmented CNN improved from **78.84% to 83.73%**, a gain of **4.89 percentage points** over the same architecture without augmentation.

MobileNetV2 also benefited from fine-tuning, improving from **79.13% to 81.24%**.

These results demonstrate that model performance depends not only on model size but also on the architecture, training strategy, augmentation, and adaptation of learned features.

---

<div align="center">

# 👨‍💻 Author

### Tushar Vala

**Data Analyst | AI/ML Developer**

<br>

### ⭐ If you found this project useful, consider giving the repository a star!

<br>

![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Powered by TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Built with Keras](https://img.shields.io/badge/Built%20with-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

---

### 🧠 From Convolution Fundamentals to Transfer Learning

**CIFAR-10 • CNN • Data Augmentation • MobileNetV2 • Fine-Tuning**

</div>
