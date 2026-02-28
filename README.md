# EEG Brain State Classifier

**Can a computer tell if you're relaxed or stressed — just from your brainwaves?**

A machine learning system that reads brainwave data from an OpenBCI EEG headset, classifies your mental state as **relaxed** or **not relaxed**, and automatically plays soothing music when you're stressed.

Built for a high school hackathon.

From Anish Saravanan, Sriya Sriram and Shriya Kanikaram.

---

## How It Works

1. An OpenBCI Cyton headset with 8 electrodes reads electrical signals from the scalp
2. The signals are broken into 5 frequency bands (delta, theta, alpha, beta, gamma) using Power Spectral Density
3. A Random Forest classifier (200 decision trees) votes on whether the brain is relaxed or not
4. When the model detects stress, calming ambient music plays automatically
5. When the brain relaxes again, the music fades out — a real-time biofeedback loop

**Key insight:** Relaxed brains produce strong alpha waves (8–13 Hz). Stressed or focused brains produce strong beta waves (13–30 Hz). The alpha/beta ratio is the single best predictor.

---

## Project Structure

```
hackathon/
├── README.md                 # This file
├── Makefile                  # Build commands (setup, run, simulate, etc.)
├── requirements.txt          # Python dependencies
├── inference.py              # Standalone live inference script (run on laptop)
├── EEG_Hackathon_Demo.ipynb  # Full Colab notebook (Parts A–F)
├── soothing_music.wav        # Auto-generated calming ambient sound (90s)
├── calm_chime.wav            # Short chime sound (2s)
├── brain_classifier_hackathon.pkl  # Trained model (auto-generated)
└── eeg_data/                 # Training CSV files (from OpenBCI)
    ├── relaxing_01.csv
    ├── relaxation_04.csv
    ├── meditation_01.csv
    ├── focus_01.csv
    ├── negative_01.csv
    ├── test_subject.csv
    └── ...
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- OpenBCI Cyton headset + Bluetooth dongle (for live demo)
- macOS, Windows, or Linux

### Setup (one time)

```bash
cd hackathon
make setup
```

This creates a virtual environment and installs all dependencies.

### Run Simulated Demo (no headset needed)

```bash
make simulate
```

Runs inference on a recorded CSV file so you can test everything without the headset.

### Run Live with OpenBCI Headset

```bash
# Mac (auto-detects serial port)
make run-mac

# Windows
make run PORT=COM5

# Linux
make run PORT=/dev/ttyUSB0

# Custom duration (default: 120 seconds)
make run-mac DURATION=180
```

### Retrain the Model

```bash
make train
```

Retrains from all CSV files in `eeg_data/` and saves a new `.pkl` model file.

### Other Commands

```bash
make check      # Verify all packages are installed
make clean      # Remove venv and cached files
make help       # Show all available commands
```

---

## The Notebook (Google Colab)

`EEG_Hackathon_Demo.ipynb` is a step-by-step walkthrough designed for 9th graders:

| Part | What It Does | Key Takeaway |
|------|-------------|--------------|
| **A** | Setup and load data | Install tools, define brainwave bands |
| **B** | Naive model (~33% accuracy) | Show why a simple approach fails |
| **C** | Improved model (80%+ accuracy) | Label fixing, augmentation, smart features |
| **D** | Simulated live demo with music | Auto-play soothing sounds when stressed |
| **E** | Real-time live demo with headset | The showstopper for judges |
| **F** | Judge Q&A preparation | 10 likely questions with confident answers |

Open it in Google Colab by uploading to Google Drive and clicking "Open with Colab."

---

## How We Improved From 33% to 80%+ Accuracy

| Problem | Fix |
|---------|-----|
| "relaxing" and "relaxation" treated as different classes | Merged into 2 clean labels: `relaxed` and `not_relaxed` |
| Only 1 training sample per 60-second file | Sliding window augmentation: 4s window, 1s step → ~56 samples per file |
| No domain-specific features | Added alpha/beta ratio — the #1 neuroscience indicator of relaxation |

---

## Model Details

- **Algorithm:** Random Forest (200 trees, max depth 15, balanced class weights)
- **Pipeline:** StandardScaler → RandomForestClassifier
- **Features:** 44 total (40 relative band powers + 4 ratio features)
- **Validation:** 5-fold stratified cross-validation
- **Training data:** ~19 CSV files → ~1,000+ augmented samples
- **Sampling rate:** 250 Hz (OpenBCI Cyton default)
- **Channels:** 8 EEG electrodes

---

## Demo Script for Judges (10 minutes)

1. **2 min** — Explain the concept (brain → headset → classifier → music)
2. **1 min** — Show the naive model failing (Part B of notebook)
3. **2 min** — Show how you fixed it with data science (Part C)
4. **3 min** — Live demo with a volunteer wearing the headset
5. **2 min** — Answer judge questions (see Part F of notebook)

### Live Demo Steps

1. Volunteer puts on the headset
2. Run `make run-mac` (or `make simulate` as backup)
3. Ask them to close their eyes and breathe slowly → model says "Relaxed"
4. Ask them to count backwards from 1000 by 7s (1000, 993, 986…) → model says "NOT RELAXED" → music starts
5. Ask them to relax again → music fades out

---

## Dependencies

- `numpy` — numerical computation
- `pandas` — data loading and manipulation
- `scipy` — signal processing (Welch PSD)
- `scikit-learn` — machine learning (Random Forest, StandardScaler)
- `matplotlib` — visualization
- `joblib` — model serialization
- `brainflow` — OpenBCI headset communication
- `pygame` — audio playback

---

## Troubleshooting

**"No eeg_data folder found"**
Copy your CSV files into an `eeg_data/` folder next to `inference.py`.

**"Serial port not found" (live demo)**
Check your port: Device Manager (Windows), `ls /dev/tty.*` (Mac), `ls /dev/ttyUSB*` (Linux). Make sure the OpenBCI GUI is closed.

**"pygame audio not working"**
On Mac: `brew install sdl2 sdl2_mixer` then reinstall pygame. On headless servers, audio won't work — use `--simulate` mode.

**Low accuracy after retraining**
Make sure CSV files are tab-separated with 8 EEG channels in columns 1–8, and filenames start with a valid label (relaxing, relaxation, meditation, focus, negative, etc.).
