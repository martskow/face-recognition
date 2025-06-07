# Face Recognition Feature Extraction Benchmark
This project is being developed as part of the "Scientific and Implementation Project" course.

This repository contains the full code and dataset for a comparative study of facial feature extraction methods, including HOG, LBP, MobileNetV3, and FaceNet. The goal is to evaluate the trade-offs between accuracy, robustness to image perturbations, and computational efficiency in the context of binary face authentication.

## Repository Structure
├── `dataset/` - Dataset (authorized/unauthorized subfolders).

├── `plots/` - Statistical visualization in the form of histograms, box plots and heatmaps.

├── `statistic_results/` - results of the statistical analysis in the form of `.csv` files.

├── `FaceBinaryDataset.py` - PyTorch Dataset class for loading and preprocessing images.

├── `README.md` - This file.

├── `build_pipeline.py` - Common pipeline: feature extraction, scaling, and classification.

├── `extraction.py` - Implementations of HOG, LBP, CNN (MobileNetV3), and FaceNet feature extractors.

├── `metrics.py` - Functions to compute FAR, FRR, precision, recall.

├── `perturbations.py` - Code to apply image perturbations.

├── `requirements.txt` - List of required Python packages.

├── `research.ipynb` - Main notebook: training, cross-validation, evaluation.

└── `results.ipynb` - Statistical analysis and visualization scripts.

## 🧠 Feature Extraction Methods

- **HOG (Histogram of Oriented Gradients)**
- **LBP (Local Binary Patterns)**
- **CNN (MobileNetV3, pretrained)**
- **FaceNet (InceptionResNetV1, pretrained)**

All methods use a shared pipeline and are evaluated with the same SVM classifier.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/martskow/face-recognition.git
cd face-recognition
```

2. **Install required packages**
   
```bash
pip install -r requirements.txt
```

## Dataset
The dataset is already included in the dataset/ directory, structured as follows:

dataset/

├── authorized/

│   ├── user_00001/

│   ├── user_00002/

│   └── ...

└── unauthorized/

    ├── user_00001/
    
    ├── user_00002/
    
    └── ...
    
Each subfolder contains face images of a single user.


## Running the Experiments

1. Preprocessing and Data Loading
Handled automatically by FaceBinaryDataset.py, which:

  - Loads face images,
  - Performs face detection and cropping using HaarCascade,
  - Resizes images to 160×160 pixels,
  - Applies transformations if specified.

2. Feature Extraction
Each extractor is implemented in extraction.py. You can use them individually or within the full pipeline.

3. Training and Evaluation
The core training and evaluation logic (with k-fold cross-validation) is inside research.ipynb. Simply run the notebook to:
  - Train an SVM on extracted features,
  - Evaluate with FAR, FRR, precision, and recall.

4. Analyze and Visualize Results
Scripts in the results/ folder perform statistical analysis and generate comparative plots (e.g., bar charts, box plots, metric tables).

## Notes
- Face detection may occasionally fail on poor-quality images (logged to console)
- All classifiers use the same training splits for fairness.

