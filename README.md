# DAI (Detect AI Involvement)
 DAI (Detect AI Involvement) is an advanced tool designed to identify and analyze the presence of AI-generated content across various mediums. With a focus on detecting AI involvement in videos (such as face-swapped deepfakes), images, audio mimicry, and text
# Deepfake Detection Pipeline

This project trains a MesoNet deep learning model to detect deepfake videos.  
It includes preprocessing, training, and real-time detection scripts.

---

## 📌 Project Structure
```
deepfake_detection/
├── data/                     # Dataset directory
│   ├── real/                 # Folder for real videos
│   ├── fake/                 # Folder for fake videos
│   └── processed/            # Extracted face images for training
├── models/
│   └── mesonet_model.py      # MesoNet model architecture
├── scripts/
│   ├── preprocess.py         # Prepares the dataset (extracts faces)
│   ├── train.py              # Trains the deepfake detection model
│   ├── detect.py             # Runs deepfake detection on a video
├── results/
│   ├── trained_model.h5      # Saved trained model
│   └── logs/                 # Training logs/metrics
├── requirements.txt          # Dependencies file
└── README.md                 # Documentation
```

---

## ✅ Installation Instructions
### Step 1: Install Dependencies
#### 🔹 Method 1: Using Virtual Environment
1. Open **PowerShell** (or Command Prompt) and navigate to the project folder:
   ```powershell
   cd path\to\deepfake_detection
   ```
2. Create a virtual environment and activate it:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

#### 🔹 Method 2: Install Globally (Without Virtual Environment)
If you don’t want a virtual environment, simply run:
```powershell
pip install -r requirements.txt
```

---

## 📥 Step 2: Download a Deepfake Dataset
You need a dataset containing real and fake videos. Here are some options:

1. **FaceForensics++** (Best Choice)  
   👉 [Download Here](https://github.com/ondyari/FaceForensics)  
   - Extract `real` and `fake` videos into `data/real/` and `data/fake/`.

2. **Kaggle Deepfake Dataset**  
   - **Sign in to Kaggle**: [https://www.kaggle.com/](https://www.kaggle.com/)
   - **Accept competition rules** (if required).
   - **Download dataset** using:
     ```powershell
     kaggle datasets download -d xhlulu/deepfake-detection-challenge-images
     ```

---

## 🛠 Step 3: Preprocess the Dataset
Extract face images from videos and store them in `data/processed/`:

```powershell
python scripts/preprocess.py
```

---

## 🎯 Step 4: Train the MesoNet Model
Train the model on preprocessed data:

```powershell
python scripts/train.py
```

- The trained model will be saved as `results/trained_model.h5`.

---

## 🎬 Step 5: Detect Deepfakes
Run the trained model on a test video:

```powershell
python scripts/detect.py
```

- The script will display the video frame-by-frame with **"Real"** or **"Fake"** labels.

---

---

## 🎉 You're All Set!
Now you can **train** and **detect deepfakes** on your own videos! 🚀  

If you have any issues, feel free to ask for help! 😊
GITHUB ID :- https://github.com/arpitchaubey
Portfolio :- https://arpitchaubay-dev.vercel.app
