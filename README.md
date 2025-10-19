# 🎙️ Speaker Recognition App

An end-to-end **Deep Learning Speaker Identification System** built with **PyTorch**, **SpeechBrain**, and **Streamlit**.  
This project identifies speakers from short audio recordings by extracting **ECAPA-TDNN embeddings**, classifying them through a **custom neural network**, and returning the **predicted speaker** in real time through an interactive web interface.

---

##  Project Overview

The objective of this project is to design a professional-grade **AI-based speaker recognition system** capable of accurately identifying speakers from voice samples.  

The system integrates advanced **feature extraction**, **deep learning classification**, and **web deployment** to demonstrate the complete lifecycle of an applied AI project — from research to production.

![App Screenshot](./screenshot%20of%20the%20web%20application%20working%20.png)

---

##  Getting Started

Follow these steps to set up and run the project locally.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ricardos-ai/speaker-recognition-app.git
cd speaker-recognition-app
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

Then open your browser at 👉 **http://localhost:8501**

---

## ⚙️ Features

- Upload `.wav`, `.mp3`, or `.ogg` audio files directly through the web interface  
- Automatic audio preprocessing (resampling, normalization, padding)  
- Extraction of **speaker embeddings** using **SpeechBrain’s ECAPA-TDNN** model  
- Classification with a **custom PyTorch neural network** trained on multiple speakers  
- Real-time prediction and visualization in a **Streamlit** web app  

---

## 🧩 Architecture Overview

```mermaid
flowchart TD
    A[Upload audio file (.wav/.mp3/.ogg)] --> B[Audio preprocessing]
    B --> C[Embedding extraction (ECAPA-TDNN)]
    C --> D[PyTorch classifier]
    D --> E[Predicted speaker output]
```

---

## Technical Details

| Component | Description |
|------------|-------------|
| **Language** | Python |
| **Frameworks** | PyTorch, SpeechBrain, Streamlit |
| **Audio Processing** | Torchaudio, Pydub |
| **Feature Extraction** | ECAPA-TDNN pretrained model |
| **Classifier** | Fully-connected PyTorch model with batch normalization |
| **Model Artifacts** | `best_speaker_classifier.pt`, `speaker_label_encoder.pkl` |
| **Interface** | Streamlit web app for real-time inference |

---

##  Project Structure

```
speaker-recognition-app/
│
├── streamlit_app.py                     # Streamlit interface
├── -- Speaker Recognition.ipynb  # Model training & experiments
├── best_speaker_classifier.pt           # Trained PyTorch model
├── speaker_label_encoder.pkl            # Label encoder
├── requirements.txt                     # Dependencies
├── README.md                            # Project documentation
└── screenshot of the web application working .png
```

---

## 🧪 Model Development

- **Baseline:** Dense neural network using averaged MFCC features.  
- **Improved version:** Replaced static MFCCs with **ECAPA-TDNN embeddings** for more robust voice representations.  
- **Classifier training:** Custom feed-forward PyTorch model fine-tuned on embeddings with cross-entropy loss.  
- **Evaluation:** Tested on multiple real speakers, achieving >95% accuracy on validation data.

---

## 📊 Evaluation & Visualization

- **PCA / t-SNE** to visualize speaker embedding clusters.  
- **Confusion matrix** for per-speaker precision.  
- **Noise robustness** tests with background and varying durations.  

---

## 💻 Deployment

The final model is deployed as a **Streamlit web app** for real-time usage:  
- Upload a voice sample → automatic preprocessing and prediction.  
- Lightweight and deployable on **Streamlit Cloud** or **Hugging Face Spaces**.  

---

## 🔬 Future Improvements

- Add speaker **verification mode** (same/different speaker)  
- Train on larger open datasets (VoxCeleb, LibriSpeech)  
- Integrate **Voice Activity Detection (VAD)** for cleaner input handling  
- Containerize the project with **Docker** for reproducibility  

---

## 🧾 Requirements

Main dependencies:

```
torch
torchaudio
speechbrain
streamlit
pydub
numpy
scikit-learn
joblib
```

Install them via:
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Ricardos Abi Akar**  
AI & Analytics Engineer — MSc in Data Science & AI Strategy @ emlyon business school | École 42  
🔗 [GitHub: ricardos-ai](https://github.com/ricardos-ai)  
💼 [LinkedIn: ricardos-abi-akar](https://linkedin.com/in/ricardos-abi-akar)

---

⭐ **If you find this project interesting, don’t forget to give it a star!**
