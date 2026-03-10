# 😊 Facial Emotion Recognition

A Convolutional Neural Network (CNN) built from scratch to detect human emotions from facial images, trained on the FER2013 dataset.

---

## 📌 Project Description

This project implements a deep learning model that identifies **7 human emotions** from grayscale facial images:

| Emotion | Emoji |
|---------|-------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprise | 😲 |

Built using **TensorFlow/Keras** and trained on the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), which contains 35,887 grayscale 48×48 pixel face images.

---

## 🧠 Model Architecture

A custom 3-block CNN architecture built from scratch — no pretrained weights.

```
Input (48x48x1 grayscale image)
        ↓
┌─────────────────────────────┐
│  Block 1                    │
│  Conv2D(32, 3x3) + BN       │
│  Conv2D(32, 3x3) + BN       │
│  MaxPooling2D + Dropout(0.25)│
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Block 2                    │
│  Conv2D(64, 3x3) + BN       │
│  Conv2D(64, 3x3) + BN       │
│  MaxPooling2D + Dropout(0.25)│
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Block 3                    │
│  Conv2D(128, 3x3) + BN      │
│  Conv2D(128, 3x3) + BN      │
│  MaxPooling2D + Dropout(0.25)│
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Fully Connected            │
│  Flatten                    │
│  Dense(256) + BN            │
│  Dropout(0.5)               │
│  Dense(7, softmax)          │
└─────────────────────────────┘
        ↓
Output (7 emotion probabilities)
```

**Key design choices:**
- **BatchNormalization** after every Conv layer for faster, stable training
- **Dropout** at 0.25 (conv blocks) and 0.5 (dense layer) to prevent overfitting
- **ReLU activations** throughout for non-linearity
- **Softmax** output for multi-class probability distribution
- Total parameters: ~1.1M

---

## 📊 Results & Accuracy

### Final Performance:

| Dataset | Accuracy | Loss |
|---------|----------|------|
| Training | 71.4% | 0.765 |
| Validation | 65.5% | 0.969 |
| **Test** | **65.67%** | **0.959** |

> 🎯 FER2013 human-level accuracy is ~65% — this model matches human performance!

### Per-class Performance (from Confusion Matrix):

| Emotion | Performance | Notes |
|---------|------------|-------|
| Happy | ⭐⭐⭐ 86% | Most distinct features |
| Surprise | ⭐⭐⭐ 78% | Wide eyes easy to detect |
| Neutral | ⭐⭐ 72% | Solid performance |
| Angry | ⭐⭐ 60% | Sometimes confused with Fear |
| Fear | ⭐⭐ 37% | Visually similar to Angry/Sad |
| Sad | ⭐⭐ 55% | Confused with Neutral |
| Disgust | ⭐ 38% | Fewest training samples (~500) |

### Training Configuration:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 64 |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

---

## 🛠️ Tech Stack

- Python 3.12
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Google Colab (T4 GPU)

---

## 📁 Project Structure

```
facial-emotion-recognition/
│
├── mood_identifier.ipynb     # Main Colab notebook
├── mood_model.h5             # Saved trained model
├── training_history.json     # Training metrics history
└── README.md
```

---

## 🔮 Future Improvements (v2)

- [ ] Keras augmentation layers built into the model
- [ ] Transfer Learning with VGG16/ResNet
- [ ] Class weights to handle class imbalance (Disgust only ~500 samples)
- [ ] Grad-CAM visualization to see what the model "looks at"
- [ ] Real-time webcam emotion detection

---

## 📚 References

- [FER2013 Dataset — Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [Deep Learning Specialization — Andrew Ng, Coursera](https://www.coursera.org/specializations/deep-learning)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
