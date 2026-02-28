#!/usr/bin/env python3
"""
============================================================
  EEG Brain State Classifier — Live Inference Script
============================================================
  Run this on your laptop to do REAL-TIME classification
  with the OpenBCI Cyton headset.

  Usage:
      python inference.py                    (uses default COM5)
      python inference.py --port COM5        (Windows)
      python inference.py --port /dev/tty.usbserial-XXXX  (Mac)
      python inference.py --port /dev/ttyUSB0              (Linux)
      python inference.py --simulate         (no headset needed)

  Before running:
      pip install numpy scipy scikit-learn brainflow pygame joblib

  Folder layout expected:
      hackathon/
      ├── inference.py          <-- this file
      ├── eeg_data/
      │   ├── relaxing_01.csv
      │   ├── focus_01.csv
      │   ├── test_subject.csv
      │   └── ... (all your training CSVs)
      └── soothing_music.wav    (auto-generated if missing)

      From: Sriy Sriram, Sriya Kanikaram, Anish Saravananan
============================================================
"""

import os
import sys
import time
import wave
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

warnings.filterwarnings("ignore")

# ============================================================
#  CONSTANTS
# ============================================================
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
SAMPLING_RATE = 250
NUM_CHANNELS = 8
NUM_BAND_FEATURES = NUM_CHANNELS * len(BANDS)  # 40
NUM_RATIO_FEATURES = 4
TOTAL_FEATURES = NUM_BAND_FEATURES + NUM_RATIO_FEATURES  # 44

LABEL_MAP = {
    "relaxing":     "relaxed",
    "relaxation":   "relaxed",
    "meditation":   "relaxed",
    "negative":     "not_relaxed",
    "focus":        "not_relaxed",
    "deepthinking": "not_relaxed",
    "stressed":     "not_relaxed",
}
CLASS_NAMES = ["relaxed", "not_relaxed"]


# ============================================================
#  PATHS — auto-detect relative to this script
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "eeg_data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "brain_classifier_hackathon.pkl")
MUSIC_FILE = os.path.join(SCRIPT_DIR, "soothing_music.wav")


# ============================================================
#  FEATURE EXTRACTION  (must match the training notebook)
# ============================================================
def extract_band_powers_one_channel(signal_segment):
    """Calculate relative power in each frequency band for ONE channel."""
    if len(signal_segment) < SAMPLING_RATE:
        return [0.2] * len(BANDS)

    nperseg = min(2 * SAMPLING_RATE, len(signal_segment))
    frequencies, power = welch(signal_segment, fs=SAMPLING_RATE, nperseg=nperseg)

    band_powers = []
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies <= high)
        bp = np.trapezoid(power[mask], frequencies[mask]) if np.any(mask) else 0.0
        band_powers.append(bp)

    total = sum(band_powers)
    if total == 0:
        return [0.2] * len(BANDS)
    return [p / total for p in band_powers]


def extract_features(eeg_window):
    """
    Extract 44 features from an 8-channel EEG window.
      - 40 relative band powers  (8 channels x 5 bands)
      -  4 ratio features        (alpha/beta, theta/beta, mean_alpha, mean_beta)
    """
    features = []
    alpha_list, beta_list, theta_list = [], [], []

    for ch in range(eeg_window.shape[0]):
        channel_data = eeg_window[ch]
        channel_data = channel_data[~np.isnan(channel_data)]
        bp = extract_band_powers_one_channel(channel_data)
        features.extend(bp)

        # Band order: delta(0), theta(1), alpha(2), beta(3), gamma(4)
        alpha_list.append(bp[2])
        beta_list.append(bp[3])
        theta_list.append(bp[1])

    avg_alpha = np.mean(alpha_list)
    avg_beta  = np.mean(beta_list) + 1e-10
    avg_theta = np.mean(theta_list)

    features.append(avg_alpha / avg_beta)   # Alpha/Beta ratio
    features.append(avg_theta / avg_beta)   # Theta/Beta ratio
    features.append(avg_alpha)              # Mean alpha power
    features.append(avg_beta)               # Mean beta power

    return features


# ============================================================
#  TRAINING — load CSVs, augment, train, save model
# ============================================================
def load_file_with_augmentation(filepath, label, window_sec=4, step_sec=1):
    """Slide a 4-second window across a CSV file → many training samples."""
    df = pd.read_csv(filepath, header=None, sep="\t")
    if df.shape[1] < 9:
        return [], []

    eeg_data = df.iloc[:, 1:9].values.T
    total_samples = eeg_data.shape[1]
    window_samples = int(window_sec * SAMPLING_RATE)
    step_samples = int(step_sec * SAMPLING_RATE)

    features_list, labels_list = [], []
    start = 0
    while start + window_samples <= total_samples:
        window = eeg_data[:, start:start + window_samples]
        feats = extract_features(window)
        features_list.append(feats)
        labels_list.append(label)
        start += step_samples

    return features_list, labels_list


def train_model():
    """Train the classifier from CSV files in eeg_data/ and save to disk."""
    print("=" * 60)
    print("  TRAINING MODEL FROM CSV FILES")
    print("=" * 60)

    if not os.path.isdir(DATA_FOLDER):
        print(f"\n  ERROR: Data folder not found: {DATA_FOLDER}")
        print(f"  Make sure 'eeg_data/' is next to this script.")
        sys.exit(1)

    all_features, all_labels = [], []

    for filename in sorted(os.listdir(DATA_FOLDER)):
        if not filename.endswith(".csv") or "test" in filename.lower():
            continue

        raw_label = filename.split("_")[0]
        mapped = LABEL_MAP.get(raw_label)
        if mapped is None:
            print(f"  Skipping unknown label '{raw_label}' in {filename}")
            continue

        filepath = os.path.join(DATA_FOLDER, filename)
        feats, labels = load_file_with_augmentation(filepath, mapped)
        all_features.extend(feats)
        all_labels.extend(labels)

        tag = "(calm)" if mapped == "relaxed" else "(active)"
        print(f"  {filename:35s} -> {mapped:12s}  ({len(feats):4d} windows) {tag}")

    if len(all_features) == 0:
        print("\n  ERROR: No training data found!")
        sys.exit(1)

    X = np.array(all_features)
    y_text = np.array(all_labels)
    label_to_num = {"relaxed": 0, "not_relaxed": 1}
    num_to_label = {0: "relaxed", 1: "not_relaxed"}
    y = np.array([label_to_num[l] for l in y_text])

    relaxed_n = np.sum(y == 0)
    stressed_n = np.sum(y == 1)
    print(f"\n  Total samples:  {len(X):,}")
    print(f"  Relaxed:        {relaxed_n:,}  ({relaxed_n/len(X)*100:.1f}%)")
    print(f"  Not Relaxed:    {stressed_n:,}  ({stressed_n/len(X)*100:.1f}%)")

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Cross-validation
    print("\n  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"  Fold scores: {[f'{s:.1%}' for s in scores]}")
    print(f"  Average:     {scores.mean():.1%} (+/- {scores.std():.1%})")

    # Final train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n  Test accuracy: {acc:.1%}")
    print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))

    # Save
    joblib.dump(
        {
            "pipeline": pipeline,
            "num_to_label": num_to_label,
            "label_to_num": label_to_num,
            "class_names": CLASS_NAMES,
            "bands": BANDS,
            "sampling_rate": SAMPLING_RATE,
            "num_channels": NUM_CHANNELS,
        },
        MODEL_PATH,
    )
    print(f"  Model saved -> {MODEL_PATH}")
    print("=" * 60)
    return pipeline, num_to_label


def load_model():
    """Load a previously trained model, or train a new one."""
    if os.path.exists(MODEL_PATH):
        print(f"  Loading saved model: {MODEL_PATH}")
        saved = joblib.load(MODEL_PATH)
        return saved["pipeline"], saved["num_to_label"]
    else:
        print("  No saved model found — training from scratch...")
        return train_model()


# ============================================================
#  SOOTHING MUSIC — generate if missing
# ============================================================
def generate_soothing_music(filepath, duration_sec=90, sample_rate=44100):
    """Create a calming ambient WAV using healing frequencies."""
    print(f"  Generating soothing music ({duration_sec}s)...")
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    signal = np.zeros_like(t)
    signal += 0.15 * np.sin(2 * np.pi * 174 * t)
    signal += 0.12 * np.sin(2 * np.pi * 285 * t)
    signal += 0.08 * np.sin(2 * np.pi * 396 * t)
    signal += 0.06 * np.sin(2 * np.pi * 176 * t)
    signal += 0.04 * np.sin(2 * np.pi * 528 * t)

    # Fade in/out
    fade = int(3 * sample_rate)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)

    # Breathing rhythm (~9 breaths/minute)
    breathing = 0.8 + 0.2 * np.sin(2 * np.pi * 0.15 * t)
    signal *= breathing

    signal = signal / np.max(np.abs(signal)) * 0.7
    signal_int = (signal * 32767).astype(np.int16)

    with wave.open(filepath, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(signal_int.tobytes())

    print(f"  Created: {filepath}")


# ============================================================
#  MUSIC PLAYER — pygame wrapper
# ============================================================
class MusicPlayer:
    """Simple wrapper around pygame.mixer for soothing music."""

    def __init__(self, music_path):
        self.ready = False
        self.playing = False
        try:
            import pygame
            self.pygame = pygame
            pygame.mixer.init()
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                self.ready = True
                print(f"  Audio ready: {music_path}")
            else:
                print(f"  Music file not found — generating...")
                generate_soothing_music(music_path)
                pygame.mixer.music.load(music_path)
                self.ready = True
        except ImportError:
            print("  pygame not installed — running without audio")
            print("  Install with: pip install pygame")
        except Exception as e:
            print(f"  Audio init error: {e}")

    def play(self):
        if self.ready and not self.playing:
            self.pygame.mixer.music.play(-1)  # loop forever
            self.playing = True

    def stop(self):
        if self.ready and self.playing:
            self.pygame.mixer.music.fadeout(1000)
            self.playing = False

    def cleanup(self):
        if self.ready:
            try:
                self.pygame.mixer.music.stop()
                self.pygame.mixer.quit()
            except:
                pass


# ============================================================
#  LIVE INFERENCE — OpenBCI Cyton via BrainFlow
# ============================================================
def run_live(port, window_sec=4, update_sec=1, duration_sec=120):
    """
    Real-time inference with the OpenBCI Cyton headset.
    Reads EEG, classifies every second, plays music when stressed.
    """
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes

    pipeline, num_to_label = load_model()
    music = MusicPlayer(MUSIC_FILE)

    print()
    print("=" * 60)
    print("  REAL-TIME BRAIN STATE CLASSIFICATION")
    print("=" * 60)
    print(f"  Port:      {port}")
    print(f"  Window:    {window_sec}s")
    print(f"  Update:    every {update_sec}s")
    print(f"  Duration:  {duration_sec}s")
    print(f"  Audio:     {'Ready' if music.ready else 'Disabled'}")
    print("=" * 60)

    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    window_samples = int(window_sec * sampling_rate)

    print(f"\n  Connecting to OpenBCI Cyton on {port}...")

    try:
        board.prepare_session()
        board.start_stream()
        print("  Connected! Streaming started.")
        print(f"  Buffering {window_sec + 1}s of data...\n")
        time.sleep(window_sec + 1)

        start_time = time.time()
        relaxed_count = 0
        total_count = 0

        print("  Classification running! Press Ctrl+C to stop.\n")
        print(f"  {'Time':>8s}  {'State':<14s}  {'Confidence':<24s}  {'Music'}")
        print(f"  {'-'*8}  {'-'*14}  {'-'*24}  {'-'*8}")

        while (time.time() - start_time) < duration_sec:
            data = board.get_current_board_data(window_samples)
            if data.shape[1] < window_samples:
                time.sleep(0.5)
                continue

            eeg_window = data[eeg_channels, :]

            # Filter: bandpass 1-45 Hz + notch 60 Hz
            for ch in range(eeg_window.shape[0]):
                DataFilter.perform_bandpass(
                    eeg_window[ch], sampling_rate,
                    1.0, 45.0, 4, FilterTypes.BUTTERWORTH.value, 0
                )
                DataFilter.perform_bandstop(
                    eeg_window[ch], sampling_rate,
                    58.0, 62.0, 4, FilterTypes.BUTTERWORTH.value, 0
                )

            features = extract_features(eeg_window)
            feature_vec = np.array(features).reshape(1, -1)
            pred_num = pipeline.predict(feature_vec)[0]
            pred_proba = pipeline.predict_proba(feature_vec)[0]
            confidence = max(pred_proba)
            label = num_to_label[pred_num]

            total_count += 1
            if label == "relaxed":
                relaxed_count += 1

            elapsed = time.time() - start_time
            bar = "#" * int(confidence * 20) + "." * (20 - int(confidence * 20))
            state_str = "Relaxed" if label == "relaxed" else "NOT RELAXED"
            music_str = ""

            if label == "not_relaxed" and not music.playing:
                music.play()
                music_str = "-> ON"
            elif label == "relaxed" and music.playing:
                music.stop()
                music_str = "-> OFF"

            print(f"  {elapsed:7.1f}s  {state_str:<14s}  [{bar}] {confidence:4.0%}   {music_str}")
            time.sleep(update_sec)

    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Is '{port}' the correct serial port?")
        print(f"    2. Is the headset powered ON?")
        print(f"    3. Is the Bluetooth dongle plugged in?")
        print(f"    4. Is the OpenBCI GUI CLOSED?")
        print(f"    5. Try: python inference.py --simulate")
    finally:
        music.stop()
        music.cleanup()
        try:
            board.stop_stream()
            board.release_session()
        except:
            pass
        if total_count > 0:
            print(f"\n  {'='*60}")
            print(f"  SESSION SUMMARY")
            print(f"  {'='*60}")
            print(f"  Total readings:  {total_count}")
            print(f"  Relaxed:         {relaxed_count}/{total_count} ({relaxed_count/total_count*100:.0f}%)")
            print(f"  Not relaxed:     {total_count-relaxed_count}/{total_count} ({(total_count-relaxed_count)/total_count*100:.0f}%)")
            print(f"  {'='*60}")
        print("  Disconnected safely.")


# ============================================================
#  SIMULATED INFERENCE — no headset needed
# ============================================================
def run_simulated(window_sec=4, step_sec=2):
    """
    Run inference on a recorded CSV file (no headset required).
    Great for testing and demos when the headset isn't available.
    """
    pipeline, num_to_label = load_model()
    music = MusicPlayer(MUSIC_FILE)

    # Find a test file
    test_file = os.path.join(DATA_FOLDER, "test_subject.csv")
    if not os.path.exists(test_file):
        # Fall back to any training CSV
        for f in sorted(os.listdir(DATA_FOLDER)):
            if f.endswith(".csv"):
                test_file = os.path.join(DATA_FOLDER, f)
                break

    if not os.path.exists(test_file):
        print("  ERROR: No CSV files found in eeg_data/")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  SIMULATED LIVE INFERENCE (from file)")
    print("=" * 60)
    print(f"  File:    {os.path.basename(test_file)}")
    print(f"  Window:  {window_sec}s")
    print(f"  Step:    {step_sec}s")
    print(f"  Audio:   {'Ready' if music.ready else 'Disabled'}")
    print("=" * 60)

    df = pd.read_csv(test_file, header=None, sep="\t")
    if df.shape[1] < 9:
        print("  ERROR: File has too few columns.")
        sys.exit(1)

    eeg_data = df.iloc[:, 1:9].values.T
    window_samples = int(window_sec * SAMPLING_RATE)
    step_samples = int(step_sec * SAMPLING_RATE)
    total_duration = eeg_data.shape[1] / SAMPLING_RATE

    print(f"  Recording length: {total_duration:.1f}s")
    print()
    print(f"  {'Time':>8s}  {'State':<14s}  {'Confidence':<24s}  {'Music'}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*24}  {'-'*8}")

    start = 0
    relaxed_count = 0
    total_count = 0

    try:
        while start + window_samples <= eeg_data.shape[1]:
            window = eeg_data[:, start : start + window_samples]
            features = extract_features(window)
            feature_vec = np.array(features).reshape(1, -1)

            pred_num = pipeline.predict(feature_vec)[0]
            pred_proba = pipeline.predict_proba(feature_vec)[0]
            confidence = max(pred_proba)
            label = num_to_label[pred_num]

            total_count += 1
            if label == "relaxed":
                relaxed_count += 1

            mid_time = (start + start + window_samples) / 2 / SAMPLING_RATE
            bar = "#" * int(confidence * 20) + "." * (20 - int(confidence * 20))
            state_str = "Relaxed" if label == "relaxed" else "NOT RELAXED"
            music_str = ""

            if label == "not_relaxed" and not music.playing:
                music.play()
                music_str = "-> ON"
            elif label == "relaxed" and music.playing:
                music.stop()
                music_str = "-> OFF"

            print(f"  {mid_time:7.1f}s  {state_str:<14s}  [{bar}] {confidence:4.0%}   {music_str}")

            # Pause briefly so judges can see it scroll in real-time
            time.sleep(0.3)

            start += step_samples

    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
    finally:
        music.stop()
        music.cleanup()

    if total_count > 0:
        print(f"\n  {'='*60}")
        print(f"  SESSION SUMMARY")
        print(f"  {'='*60}")
        print(f"  File:            {os.path.basename(test_file)}")
        print(f"  Total readings:  {total_count}")
        print(f"  Relaxed:         {relaxed_count}/{total_count} ({relaxed_count/total_count*100:.0f}%)")
        print(f"  Not relaxed:     {total_count-relaxed_count}/{total_count} ({(total_count-relaxed_count)/total_count*100:.0f}%)")
        print(f"  {'='*60}")


# ============================================================
#  MAIN — argument parsing
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="EEG Brain State Classifier — Live Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --simulate              Test without headset
  python inference.py --port COM5             Windows + OpenBCI
  python inference.py --port /dev/ttyUSB0     Linux + OpenBCI
  python inference.py --train                 Retrain model only
        """,
    )
    parser.add_argument(
        "--port", type=str, default="COM5",
        help="Serial port for OpenBCI (default: COM5)",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run on a recorded CSV file (no headset needed)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train/retrain the model and exit",
    )
    parser.add_argument(
        "--window", type=int, default=4,
        help="Classification window in seconds (default: 4)",
    )
    parser.add_argument(
        "--duration", type=int, default=120,
        help="Live session duration in seconds (default: 120)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  EEG Brain State Classifier")
    print("  Hackathon Live Inference Script")
    print("=" * 60)

    # Generate music if missing
    if not os.path.exists(MUSIC_FILE):
        generate_soothing_music(MUSIC_FILE)

    # Train-only mode
    if args.train:
        train_model()
        return

    # Make sure model exists
    if not os.path.exists(MODEL_PATH):
        print("\n  No trained model found — training now...\n")
        train_model()

    # Run inference
    if args.simulate:
        run_simulated(window_sec=args.window, step_sec=2)
    else:
        run_live(
            port=args.port,
            window_sec=args.window,
            update_sec=1,
            duration_sec=args.duration,
        )

    print("\n  Done! Good luck at the hackathon!\n")


if __name__ == "__main__":
    main()
