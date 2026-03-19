"""
Download and process the MAFAT Radar Challenge dataset for use with LinOSS.

The MAFAT Radar Challenge is a binary classification task (human vs. animal)
using raw I/Q radar data. Each segment is a 128x32 complex-valued matrix
(128 range bins x 32 slow-time pulses).

Dataset source: https://competitions.codalab.org/competitions/25389
Reference implementation: https://github.com/expectopatronm/MAFAT-RADAR-Challenge

Two processing modes:
  --mode iq    : Raw I/Q → average across range bins → (32, 2) real/imag channels
  --mode fft   : Hann window → FFT → log-magnitude → normalize → (126, 32) spectrogram

Usage:
    # Process real data (download .pkl + .csv files first):
    python -m data_dir.process_mafat --mode iq

    # Generate synthetic data for quick testing:
    python -m data_dir.process_mafat --synthetic

    # Use FFT spectrogram mode (richer features):
    python -m data_dir.process_mafat --mode fft
"""

import os
import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Data loading (follows the original MAFAT challenge repo structure)
# ---------------------------------------------------------------------------

def load_data(file_path):
    """Load MAFAT data by combining .pkl (I/Q data) and .csv (metadata).

    Mirrors the loading approach from the original challenge repo:
    https://github.com/expectopatronm/MAFAT-RADAR-Challenge/blob/master/my_utils/dataloader.py

    Args:
        file_path: Base path without extension (e.g. "data_dir/raw/MAFAT/Training").
                   Expects both file_path.pkl and file_path.csv to exist.

    Returns:
        Dictionary with all fields as numpy arrays.
    """
    pkl_path = file_path + ".pkl"
    csv_path = file_path + ".csv"

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    if os.path.exists(csv_path):
        csv_data = pd.read_csv(csv_path)
        data_dict = {**csv_data.to_dict(orient="list"), **pkl_data}
    else:
        data_dict = pkl_data

    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    return data_dict


# ---------------------------------------------------------------------------
# Preprocessing (from the original challenge repo's pre_processor.py)
# ---------------------------------------------------------------------------

def hann(iq, window=None):
    """Apply Hann window to I/Q data along the slow-time axis."""
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1]).reshape(-1, 1)
    hann_col = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hann_col * iq[window[0]:window[1]])[1:-1]


def fft(iq, axis=0):
    """Apply Hann window, FFT, then log-magnitude to I/Q data."""
    return np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))


def max_value_on_doppler(iq, doppler_burst):
    """Mark Doppler peak locations with maximum value in the spectrogram."""
    iq_max_value = np.max(iq)
    for i in range(iq.shape[1]):
        if doppler_burst[i] >= len(iq):
            continue
        iq[doppler_burst[i], i] = iq_max_value
    return iq


def normalize(iq):
    """Zero-mean, unit-variance normalization."""
    m = iq.mean()
    s = iq.std()
    if s == 0:
        return iq - m
    return (iq - m) / s


def data_preprocess_fft(data):
    """Full FFT-based preprocessing from the original MAFAT challenge.

    Pipeline per segment: Hann window → FFT → log|.| → Doppler marking → normalize.
    Input iq_sweep_burst shape: (n_segments, 128, 32) complex
    Output shape: (n_segments, 126, 32) real (Hann trims 2 rows)
    """
    X = []
    for i in range(len(data["iq_sweep_burst"])):
        iq = fft(data["iq_sweep_burst"][i])
        if "doppler_burst" in data:
            iq = max_value_on_doppler(iq, data["doppler_burst"][i])
        iq = normalize(iq)
        X.append(iq)
    return np.array(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# Download instructions
# ---------------------------------------------------------------------------

def ensure_mafat_data(raw_dir):
    """Check that raw MAFAT data files exist, print download instructions if not."""
    os.makedirs(raw_dir, exist_ok=True)

    train_pkl = os.path.join(raw_dir, "MAFAT RADAR Challenge - Training Set V1.pkl")
    aux_pkl = os.path.join(raw_dir, "MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl")

    if os.path.exists(train_pkl) and os.path.exists(aux_pkl):
        print("MAFAT data files found.")
        return

    print("=" * 70)
    print("MAFAT Radar Challenge Dataset - Manual Download Required")
    print("=" * 70)
    print()
    print("The MAFAT dataset requires manual download from the competition page.")
    print()
    print("Steps:")
    print("1. Go to: https://competitions.codalab.org/competitions/25389#participate")
    print("2. Download the following file pairs (.pkl + .csv each):")
    print("   - MAFAT RADAR Challenge - Training Set V1")
    print("   - MAFAT RADAR Challenge - Auxiliary Experiment Set V2")
    print(f"3. Place all downloaded files in: {os.path.abspath(raw_dir)}")
    print()
    print("Alternatively, generate synthetic data for testing:")
    print("   python -m data_dir.process_mafat --synthetic")
    print("=" * 70)
    raise FileNotFoundError(
        f"MAFAT data files not found in {raw_dir}. "
        "Please download them manually (see instructions above)."
    )


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def process_mafat_iq(raw_dir, save_dir):
    """Process MAFAT data in raw I/Q mode.

    Averages across range bins to produce (n_samples, 32, 2) where the
    2 channels are real and imaginary parts of the complex I/Q signal.
    This is the simplest representation that maps directly to
    (batch, seq_len, features) for LinOSS.
    """
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "data.pkl")):
        print("Processed MAFAT I/Q data already exists, skipping.")
        return

    train_base = os.path.join(raw_dir, "MAFAT RADAR Challenge - Training Set V1")
    aux_base = os.path.join(raw_dir, "MAFAT RADAR Challenge - Auxiliary Experiment Set V2")

    all_data = []
    all_labels = []

    for base_path in [train_base, aux_base]:
        if not os.path.exists(base_path + ".pkl"):
            print(f"Warning: {base_path}.pkl not found, skipping.")
            continue

        print(f"Loading {os.path.basename(base_path)}...")
        data_dict = load_data(base_path)

        iq_sweep = data_dict["iq_sweep_burst"]  # (n_segments, 128, 32) complex
        labels = data_dict["target_type"]  # string or int labels
        n_segments = iq_sweep.shape[0]

        for i in range(n_segments):
            segment = iq_sweep[i]  # (128, 32) complex

            # Average across range bins (axis=0) → (32,) complex slow-time signal
            mean_signal = np.mean(segment, axis=0)  # (32,) complex

            iq_pair = np.stack(
                [np.real(mean_signal), np.imag(mean_signal)], axis=-1
            )  # (32, 2)
            all_data.append(iq_pair)

            # Handle string or int labels
            label = labels[i]
            if isinstance(label, str):
                all_labels.append(1 if label == "human" else 0)
            else:
                all_labels.append(int(label))

    if len(all_data) == 0:
        raise FileNotFoundError("No MAFAT data files found. Cannot process.")

    data = np.array(all_data, dtype=np.float32)  # (N, 32, 2)
    labels = np.array(all_labels, dtype=np.int32)  # (N,)

    # Normalize to [-1, 1]
    data_max = np.abs(data).max()
    if data_max > 0:
        data = data / data_max

    data = jnp.array(data)
    labels = jnp.array(labels)

    save_pickle(data, os.path.join(save_dir, "data.pkl"))
    save_pickle(labels, os.path.join(save_dir, "labels.pkl"))

    print(f"Processed MAFAT I/Q data: {data.shape}")
    print(f"  Labels: {dict(zip(*np.unique(np.array(labels), return_counts=True)))}")
    print(f"  Saved to: {save_dir}")


def process_mafat_fft(raw_dir, save_dir):
    """Process MAFAT data in FFT spectrogram mode.

    Applies the original challenge preprocessing (Hann → FFT → log|.| → normalize)
    to produce (n_samples, 126, 32) real-valued spectrograms. The sequence length
    is 126 (128 range bins minus 2 from the Hann window trim) and each timestep
    has 32 features (slow-time frequency bins).
    """
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "data.pkl")):
        print("Processed MAFAT FFT data already exists, skipping.")
        return

    train_base = os.path.join(raw_dir, "MAFAT RADAR Challenge - Training Set V1")
    aux_base = os.path.join(raw_dir, "MAFAT RADAR Challenge - Auxiliary Experiment Set V2")

    all_data = []
    all_labels = []

    for base_path in [train_base, aux_base]:
        if not os.path.exists(base_path + ".pkl"):
            print(f"Warning: {base_path}.pkl not found, skipping.")
            continue

        print(f"Loading {os.path.basename(base_path)}...")
        data_dict = load_data(base_path)

        # Apply full FFT preprocessing
        processed = data_preprocess_fft(data_dict)  # (n_segments, 126, 32)
        labels = data_dict["target_type"]

        all_data.append(processed)

        # Handle string or int labels
        seg_labels = []
        for label in labels:
            if isinstance(label, str):
                seg_labels.append(1 if label == "human" else 0)
            else:
                seg_labels.append(int(label))
        all_labels.extend(seg_labels)

    if len(all_data) == 0:
        raise FileNotFoundError("No MAFAT data files found. Cannot process.")

    data = np.concatenate(all_data, axis=0).astype(np.float32)  # (N, 126, 32)
    labels = np.array(all_labels, dtype=np.int32)

    data = jnp.array(data)
    labels = jnp.array(labels)

    save_pickle(data, os.path.join(save_dir, "data.pkl"))
    save_pickle(labels, os.path.join(save_dir, "labels.pkl"))

    print(f"Processed MAFAT FFT data: {data.shape}")
    print(f"  Labels: {dict(zip(*np.unique(np.array(labels), return_counts=True)))}")
    print(f"  Saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Synthetic data for testing
# ---------------------------------------------------------------------------

def generate_synthetic_mafat_data(save_dir, n_samples=3000, seq_len=32, seed=42):
    """Generate synthetic radar-like I/Q data for testing the pipeline.

    Creates (n_samples, seq_len, 2) data with binary labels.
    Humans get a higher-frequency micro-Doppler signature than animals.
    """
    rng = np.random.RandomState(seed)

    n_human = n_samples // 2
    n_animal = n_samples - n_human
    t = np.linspace(0, 1, seq_len)

    # Human: higher frequency micro-Doppler
    human_real = np.sin(2 * np.pi * 5 * t[None, :]) + 0.3 * rng.randn(n_human, seq_len)
    human_imag = np.cos(2 * np.pi * 5 * t[None, :]) + 0.3 * rng.randn(n_human, seq_len)
    human_iq = np.stack([human_real, human_imag], axis=-1)
    human_labels = np.ones(n_human, dtype=np.int32)

    # Animal: lower frequency pattern
    animal_real = np.sin(2 * np.pi * 1.5 * t[None, :]) + 0.3 * rng.randn(n_animal, seq_len)
    animal_imag = np.cos(2 * np.pi * 1.5 * t[None, :]) + 0.3 * rng.randn(n_animal, seq_len)
    animal_iq = np.stack([animal_real, animal_imag], axis=-1)
    animal_labels = np.zeros(n_animal, dtype=np.int32)

    data = np.concatenate([human_iq, animal_iq], axis=0)
    labels = np.concatenate([human_labels, animal_labels], axis=0)

    perm = rng.permutation(len(data))
    data = data[perm]
    labels = labels[perm]

    data_max = np.abs(data).max()
    if data_max > 0:
        data = data / data_max

    data = jnp.array(data, dtype=jnp.float32)
    labels = jnp.array(labels, dtype=jnp.int32)

    os.makedirs(save_dir, exist_ok=True)
    save_pickle(data, os.path.join(save_dir, "data.pkl"))
    save_pickle(labels, os.path.join(save_dir, "labels.pkl"))
    print(f"Saved synthetic MAFAT data: {data.shape}, labels: {labels.shape}")
    print(f"  -> {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process MAFAT Radar Challenge data")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data for testing instead of using real data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="iq",
        choices=["iq", "fft"],
        help="Processing mode: 'iq' for raw I/Q (32, 2), 'fft' for spectrogram (126, 32)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_dir",
        help="Root data directory",
    )
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw", "MAFAT")
    save_dir = os.path.join(args.data_dir, "processed", "MAFAT", "radar")

    if args.synthetic:
        generate_synthetic_mafat_data(save_dir)
    else:
        ensure_mafat_data(raw_dir)
        if args.mode == "iq":
            process_mafat_iq(raw_dir, save_dir)
        elif args.mode == "fft":
            process_mafat_fft(raw_dir, save_dir)
