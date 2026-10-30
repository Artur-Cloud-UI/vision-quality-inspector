# 👁️ Industrial Vision Quality Inspector

Automated industrial anomaly detection system built using modern Computer Vision and MLOps principles. The project leverages deep feature extraction and memory bank algorithms to detect surface defects (such as scratches and structural flaws) on manufactured parts while ignoring minor positional shifts.

## 🚀 Key Features
* **State-of-the-Art Anomaly Detection:** Implements **PatchCore** algorithm via the `Anomalib` framework.
* **Memory Bank Architecture:** Learns the structural normal pattern solely from defect-free samples (`train/good`).
* **High Precision:** Achieves exceptional performance metrics on industrial datasets.
* **Automated Inference & Heatmaps:** Generates precise anomaly localization maps (heatmaps) highlighting defective regions.

## 📊 Results & Performance
Evaluated on the industrial test benchmark:
* **Image AUROC:** `~0.99`
* **F1-Score:** `~0.93`

## 🛠️ Tech Stack
* **Python** (3.10+)
* **PyTorch** & **PyTorch Lightning**
* **Anomalib** (Intel's Library for Industrial Anomaly Detection)
* **Git / MLOps**

## 📂 Project Structure
```text
vision-quality-inspector/
│
├── data/                  # Dataset directory (MVTec AD - metal_nut)
│   ├── train/good/        # Defect-free samples used for training
│   └── test/              # Defective & normal samples used for evaluation
│
├── src/
│   ├── train.py           # Script to build the PatchCore memory bank
│   └── predict.py         # Inference script & metrics evaluation
│
├── .gitignore             # Ignored files (checkpoints, weights, results)
└── README.md              # Project documentation

How to Run

Clone the repository:

    ```bash
    git clone [https://github.com/Artur-Cloud-UI/vision-quality-inspector.git](https://github.com/Artur-Cloud-UI/vision-quality-inspector.git)
    cd vision-quality-inspector

Set up virtual environment & install dependencies:

    Bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    pip install -r requirements.txt
    
Run Training (Build Memory Bank):

    Bash
    python src/train.py

Run Inference & Evaluation:

    Bash
    python src/predict.py

License
This project is open-source and available under the MIT License.