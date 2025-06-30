# 🔍 Real-Time Violence Detection using MobileNetV2 + BiLSTM

This project is a deep learning-based violence detection system that automatically classifies whether a video sequence contains **Violence** or **NonViolence**. The model leverages **MobileNetV2** for spatial feature extraction and **Bidirectional LSTM** for temporal analysis, providing accurate, lightweight, and efficient real-time predictions.

---

## 📌 Key Features

- ✅ Real-time violence detection from videos
- 📽️ Frame-wise analysis using deep learning
- 🧠 Transfer learning with MobileNetV2
- 🔄 Sequence learning with BiLSTM (forward & backward LSTM)
- 📊 Visualizations: accuracy/loss graphs, confusion matrix, and classification report
- 🔧 Works seamlessly on Jupyter / Kaggle notebooks

---
## 📁 Dataset

- **Name**: Real-Life Violence Situations Dataset
- **Source**: Kaggle
- **Classes**: `Violence`, `NonViolence`
- **Format**: MP4 videos

[Kaggle Dataset Link](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

---

## 🧱 Model Architecture


Video Frames
    ↓
TimeDistributed MobileNetV2 (CNN for feature extraction)
    ↓
Flatten Layers
    ↓
Bidirectional LSTM (Temporal feature analysis)
    ↓
Dense Layers with Dropout (Regularization)
    ↓
Softmax (Prediction: Violence / NonViolence)

---

## 🛠️ Technologies Used

- Python  
- TensorFlow  
- Keras  
- OpenCV  
- Matplotlib  
- Scikit-learn  
- Kaggle Notebook  

---

## ⚙️ System Requirements

| Component  | Requirement                |
|------------|----------------------------|
| Processor  | Intel i5 / i7 or equivalent |
| RAM        | 8 GB minimum (16 GB preferred) |
| Disk Space | 10 GB (dataset & model)    |
| GPU        | NVIDIA (Recommended)       |
| OS         | Windows / Ubuntu / Linux   |
| IDE        | Kaggle Notebook / Jupyter  |

---

## 📊 Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 92%    |
| Precision | 91%    |
| Recall    | 90%    |
| F1 Score  | 90%    |

