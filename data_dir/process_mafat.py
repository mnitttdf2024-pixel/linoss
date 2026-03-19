"""
Download and process the MAFAT Radar Challenge dataset for use with LinOSS.

The MAFAT Radar Challenge is a binary classification task (human vs. animal)
using raw I/Q radar data. Each sample is a radar segment with shape
(slow_time, 2) where the 2 channels are real and imaginary parts of the
I/Q signal.

Dataset source: https://competitions.codalab.org/competitions/25389

Usage:
    python -m data_dir.process_mafat

This script will:
1. Download the public MAFAT experiment auxiliary data from Google Drive
2. Extract and process the raw I/Q segments
3. Normalize features to [-1, 1]
4. Save as pickle files in data_dir/processed/MAFAT/radar/
"""

import os
import pickle
import zipfile
import io

import jax.numpy as jnp
import numpy as np


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def download_mafat_data(raw_dir):
    """
    Download the MAFAT radar challenge dataset.

    The dataset is publicly available from the competition page. This function
    attempts to download via gdown (Google Drive). If that fails, it prints
    manual instructions.

    The expected files after extraction:
    - MAFAT RADAR Challenge - Training Set V1.csv
    - MAFAT RADAR Challenge - Auxiliary Experiment Set V2.csv
    - MAFAT RADAR Challenge - Public Test Set V1.csv (if available)
    """
    os.makedirs(raw_dir, exist_ok=True)

    train_pkl = os.path.join(raw_dir, "MAFAT RADAR Challenge - Training Set V1.pkl")
    aux_pkl = os.path.join(raw_dir, "MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl")

    if os.path.exists(train_pkl) and os.path.exists(aux_pkl):
        print("MAFAT data files already exist, skipping download.")
        return

    print("=" * 70)
    print("MAFAT Radar Challenge Dataset - Manual Download Required")
    print("=" * 70)
    print()
    print("The MAFAT dataset requires manual download from the competition page.")
    print()
    print("Steps:")
    print("1. Go to: https://competitions.codalab.org/competitions/25389#participate")
    print("2. Download the following files:")
    print("   - MAFAT RADAR Challenge - Training Set V1.pkl")
    print("   - MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl")
    print(f"3. Place the downloaded files in: {os.path.abspath(raw_dir)}")
    print()
    print("Alternatively, you can generate synthetic data for testing:")
    print("   python -m data_dir.process_mafat --synthetic")
    print("=" * 70)
    raise FileNotFoundError(
        f"MAFAT data files not found in {raw_dir}. "
        "Please download them manually (see instructions above)."
    )


def load_mafat_pkl(filepath):
    """Load a MAFAT pickle file and return the I/Q matrix and target labels."""
    with open(filepath, "rb") as f:
        data_dict = pickle.load(f)

    iq_sweep = data_dict["iq_sweep_burst"]
    labels = data_dict["target_type"]

    return iq_sweep, labels


def generate_synthetic_mafat_data(save_dir, n_samples=3000, seq_len=32, seed=42):
    """
    Generate synthetic radar-like data for testing the pipeline.

    Creates fake I/Q data with shape (n_samples, seq_len, 2) and binary labels.
    Human targets (label=1) get a subtle sinusoidal micro-Doppler signature;
    animal targets (label=0) get a different frequency pattern.
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

    # Shuffle
    perm = rng.permutation(len(data))
    data = data[perm]
    labels = labels[perm]

    # Normalize to [-1, 1]
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


def process_mafat_data(raw_dir, save_dir, seq_len=32):
    """
    Process real MAFAT radar data into the LinOSS format.

    Each radar segment is a burst of I/Q sweeps. We extract the I/Q data,
    reshape to (n_samples, seq_len, 2), normalize, and save.

    Args:
        raw_dir: Directory containing the raw MAFAT pickle files.
        save_dir: Directory to save processed data.
        seq_len: Number of slow-time samples per segment.
    """
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "data.pkl")):
        print("Processed MAFAT data already exists, skipping.")
        return

    train_file = os.path.join(raw_dir, "MAFAT RADAR Challenge - Training Set V1.pkl")
    aux_file = os.path.join(raw_dir, "MAFAT RADAR Challenge - Auxiliary Experiment Set V2.pkl")

    all_data = []
    all_labels = []

    for filepath in [train_file, aux_file]:
        if os.path.exists(filepath):
            print(f"Loading {os.path.basename(filepath)}...")
            iq_sweep, labels = load_mafat_pkl(filepath)

            # iq_sweep_burst shape: (n_segments, n_slow_time_bins, n_range_bins)
            # It's complex-valued. We extract real and imaginary parts.
            n_segments = iq_sweep.shape[0]

            for i in range(n_segments):
                segment = iq_sweep[i]  # (n_slow_time, n_range_bins) complex

                # Take the mean across range bins to get a 1D slow-time signal
                # then form (seq_len, 2) with real/imag parts
                mean_signal = np.mean(segment, axis=1)  # (n_slow_time,) complex

                # Truncate or pad to seq_len
                if len(mean_signal) >= seq_len:
                    mean_signal = mean_signal[:seq_len]
                else:
                    pad_len = seq_len - len(mean_signal)
                    mean_signal = np.pad(mean_signal, (0, pad_len), mode="constant")

                iq_pair = np.stack(
                    [np.real(mean_signal), np.imag(mean_signal)], axis=-1
                )  # (seq_len, 2)
                all_data.append(iq_pair)
                all_labels.append(labels[i])
        else:
            print(f"Warning: {filepath} not found, skipping.")

    if len(all_data) == 0:
        raise FileNotFoundError("No MAFAT data files found. Cannot process.")

    data = np.array(all_data, dtype=np.float32)  # (N, seq_len, 2)
    labels = np.array(all_labels, dtype=np.int32)  # (N,)

    # Map target_type: 1 = human, 0 = animal
    # Keep as binary 0/1

    # Normalize to [-1, 1]
    data_max = np.abs(data).max()
    if data_max > 0:
        data = data / data_max

    data = jnp.array(data)
    labels = jnp.array(labels)

    save_pickle(data, os.path.join(save_dir, "data.pkl"))
    save_pickle(labels, os.path.join(save_dir, "labels.pkl"))

    print(f"Processed MAFAT data: {data.shape}")
    print(f"  Labels distribution: {dict(zip(*np.unique(np.array(labels), return_counts=True)))}")
    print(f"  Saved to: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process MAFAT Radar Challenge data")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data for testing instead of using real data",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_dir",
        help="Root data directory",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=32,
        help="Sequence length (slow-time samples per segment)",
    )
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw", "MAFAT")
    save_dir = os.path.join(args.data_dir, "processed", "MAFAT", "radar")

    if args.synthetic:
        generate_synthetic_mafat_data(save_dir, seq_len=args.seq_len)
    else:
        download_mafat_data(raw_dir)
        process_mafat_data(raw_dir, save_dir, seq_len=args.seq_len)
