# Facial Video-based Non-contact Stress Recognition Utilizing Multi-Task Learning with Peak Attention
Juncong Xu, Cheng Song, Zijie Yue, and Shuai Ding
## Citation
If you find our research useful, please consider citing:
```bibtex
@article{xu2024facial,
  title={Facial Video-Based Non-Contact Stress Recognition Utilizing Multi-Task Learning With Peak Attention},
  author={Xu, Juncong and Song, Cheng and Yue, Zijie and Ding, Shuai},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}

# Project Onboarding Guide

## 1. Prerequisites

* Python 3.8+ installed
* Git installed (if cloning from repository)
* (Optional) Conda or `venv` for virtual environments

---

## 2. Setup Instructions

### Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### Step 2: Create and activate a virtual environment

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows
```

(If you’re using Conda, adapt accordingly.)

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Running the Pipeline

### 3.1 Extract rPPG signals from videos

```bash
python pos_rppg_faceboxes.py
```

### 3.2 Preprocess extracted signals

```bash
python data_preprocess.py
```

### 3.3 Train models on preprocessed data

```bash
python train_state.py
```

### 3.4 Run prediction on a new video

```bash
python predict_video_from_batches.py
```

---

## 4. Tips

* Always activate the virtual environment before running scripts.
* Check logs or error messages in the terminal for troubleshooting.
