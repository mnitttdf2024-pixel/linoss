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
    """Load a MAFAT data file (pickle format, with or without extension).

    Also loads the companion .csv metadata file if it exists.

    Mirrors the loading approach from the original challenge repo:
    https://github.com/expectopatronm/MAFAT-RADAR-Challenge/blob/master/my_utils/dataloader.py

    Args:
        file_path: Path to the data file. Can be:
                   - With extension: "data_dir/raw/MAFAT/training_set.pkl"
                   - Without extension: "data_dir/raw/MAFAT/training_set"
                   The .csv companion is loaded automatically if present.

    Returns:
        Dictionary with all fields as numpy arrays.
    """
    # Resolve the actual pickle file path
    if os.path.exists(file_path):
        pkl_path = file_path
    elif os.path.exists(file_path + ".pkl"):
        pkl_path = file_path + ".pkl"
    elif os.path.exists(file_path + ".db"):
        pkl_path = file_path + ".db"
    else:
        raise FileNotFoundError(f"Cannot find data file: {file_path}, {file_path}.pkl, or {file_path}.db")

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    # Try to load companion CSV metadata
    base_no_ext = os.path.splitext(pkl_path)[0] if pkl_path.endswith((".pkl", ".db")) else pkl_path
    csv_path = base_no_ext + ".csv"
    if os.path.exists(csv_path):
        csv_data = pd.read_csv(csv_path)
        data_dict = {**csv_data.to_dict(orient="list"), **pkl_data}
    else:
        data_dict = pkl_data

    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    return data_dict


def find_mafat_files(raw_dir):
    """Auto-discover MAFAT data files in the raw directory.

    Supports multiple naming conventions:
    - Competition download: training_set, mini_training_set
    - Old naming: MAFAT RADAR Challenge - Training Set V1.pkl, etc.
    - Any .pkl file in the directory

    Returns:
        List of file paths (without extension) that were found.
    """
    known_names = [
        # Competition zip structure (what users actually download)
        "training_set",
        "mini_training_set",
        # Auxiliary/test sets
        "auxiliary_experiment_set",
        "synthetic_set",
        "public_test_set",
        # Old long-form names from some repos
        "MAFAT RADAR Challenge - Training Set V1",
        "MAFAT RADAR Challenge - Auxiliary Experiment Set V2",
        "MAFAT RADAR Challenge - Auxiliary Synthetic Set V2",
        "MAFAT RADAR Challenge - FULL Public Test Set V1",
    ]

    found = []
    for name in known_names:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path) or os.path.exists(path + ".pkl") or os.path.exists(path + ".db"):
            found.append(path)

    # Also scan for any .pkl or .db files not in the known list
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".pkl") or f.endswith(".db"):
                base = os.path.join(raw_dir, os.path.splitext(f)[0])
                if base not in found:
                    found.append(base)
            elif not f.endswith(".csv") and os.path.isfile(os.path.join(raw_dir, f)):
                # Files without extension (like "training_set")
                full = os.path.join(raw_dir, f)
                if full not in found:
                    # Quick check: try to open as pickle
                    try:
                        with open(full, "rb") as fh:
                            pickle.load(fh)
                        found.append(full)
                    except Exception:
                        pass

    return found


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

    found = find_mafat_files(raw_dir)
    if found:
        print(f"MAFAT data files found: {[os.path.basename(f) for f in found]}")
        return found

    print("=" * 70)
    print("MAFAT Radar Challenge Dataset - No data files found")
    print("=" * 70)
    print()
    print(f"Place your data files in: {os.path.abspath(raw_dir)}")
    print()
    print("Expected files from the competition download:")
    print("  train_data_for_competition.zip containing:")
    print("    - training_set          (main training data, ~17GB)")
    print("    - mini_training_set     (smaller subset, ~1.6GB)")
    print()
    print("Steps:")
    print("  1. Extract the zip file")
    print("  2. Copy training_set and/or mini_training_set into the directory above")
    print("  3. Re-run this script")
    print()
    print("For immediate testing, use synthetic data instead:")
    print("  python -m data_dir.process_mafat --synthetic")
    print("=" * 70)
    raise FileNotFoundError(
        f"MAFAT data files not found in {raw_dir}. "
        "See instructions above or use --synthetic for testing."
    )


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def _extract_labels(data_dict):
    """Extract binary labels from a MAFAT data dictionary.

    Handles both string labels ('human'/'animal') and integer labels (1/0).
    """
    labels = data_dict["target_type"]
    result = []
    for label in labels:
        if isinstance(label, str):
            result.append(1 if label == "human" else 0)
        else:
            result.append(int(label))
    return result


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

    data_files = find_mafat_files(raw_dir)
    if not data_files:
        raise FileNotFoundError(f"No MAFAT data files found in {raw_dir}")

    all_data = []
    all_labels = []

    for file_path in data_files:
        print(f"Loading {os.path.basename(file_path)}...")
        data_dict = load_data(file_path)

        if "iq_sweep_burst" not in data_dict:
            print(f"  Warning: no 'iq_sweep_burst' key found, skipping.")
            continue
        if "target_type" not in data_dict:
            print(f"  Warning: no 'target_type' key found (unlabeled data), skipping.")
            continue

        iq_sweep = data_dict["iq_sweep_burst"]  # (n_segments, 128, 32) complex
        print(f"  Shape: {iq_sweep.shape}, dtype: {iq_sweep.dtype}")

        # Average across range bins (axis=1) → (n_segments, 32) complex
        mean_signals = np.mean(iq_sweep, axis=1)
        # Stack real/imag → (n_segments, 32, 2)
        iq_pairs = np.stack(
            [np.real(mean_signals), np.imag(mean_signals)], axis=-1
        ).astype(np.float32)

        all_data.append(iq_pairs)
        all_labels.extend(_extract_labels(data_dict))

    if len(all_data) == 0:
        raise FileNotFoundError("No valid MAFAT data files found. Cannot process.")

    data = np.concatenate(all_data, axis=0)  # (N, 32, 2)
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

    data_files = find_mafat_files(raw_dir)
    if not data_files:
        raise FileNotFoundError(f"No MAFAT data files found in {raw_dir}")

    all_data = []
    all_labels = []

    for file_path in data_files:
        print(f"Loading {os.path.basename(file_path)}...")
        data_dict = load_data(file_path)

        if "iq_sweep_burst" not in data_dict:
            print(f"  Warning: no 'iq_sweep_burst' key found, skipping.")
            continue
        if "target_type" not in data_dict:
            print(f"  Warning: no 'target_type' key found (unlabeled data), skipping.")
            continue

        print(f"  Shape: {data_dict['iq_sweep_burst'].shape}")

        # Apply full FFT preprocessing
        processed = data_preprocess_fft(data_dict)  # (n_segments, 126, 32)
        all_data.append(processed)
        all_labels.extend(_extract_labels(data_dict))

    if len(all_data) == 0:
        raise FileNotFoundError("No valid MAFAT data files found. Cannot process.")

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

def _generate_synthetic_iq_segments(n_samples, rng, target_type="human"):
    """Generate synthetic 128x32 complex I/Q radar segments.

    Mimics the real MAFAT data structure:
    - 128 range bins (fast-time / range-velocity axis)
    - 32 slow-time pulses (pulse repetition axis)
    - Complex-valued I/Q signal

    Based on micro-Doppler radar literature:
    - Human bipedal stride rate ~0.8-1.2 Hz with harmonics up to ~5 Hz.
      Max limb (toe) velocity ~4.5 m/s. Arm swing adds extra harmonics.
      Wider range spread due to upright posture (~1.7m height).
    - Animal quadruped stride rate ~1.5-3.0 Hz with harmonics up to ~8 Hz.
      Four-leg coordination creates denser but lower-amplitude harmonics.
      Narrower range spread due to lower body profile.

    Key real-world challenge from MAFAT competition: most animals were
    recorded at low SNR and most humans at high SNR, creating a spurious
    correlation. We replicate this bias (humans tend higher SNR, animals
    tend lower) while still allowing overlap, so the model must learn
    actual micro-Doppler features rather than just noise level.

    References:
    - Chen, V.C. "The Micro-Doppler Effect in Radar"
    - MAFAT Radar Challenge (CodaLab 25389)
    - PMC 7506689: limb micro-Doppler at mm-wave
    - PMC 9105660: pedestrian/animal classification
    """
    n_range = 128
    n_slow = 32
    segments = []

    slow_t = np.linspace(0, 1, n_slow)
    range_bins = np.arange(n_range)

    for _ in range(n_samples):
        # Target signal centered at a random range bin
        center_bin = rng.randint(30, 100)

        if target_type == "human":
            # Bipedal gait: stride ~0.8-1.2 Hz, micro-Doppler harmonics up to ~5 Hz
            stride_freq = rng.uniform(0.8, 1.2)
            # Humans have 2-4 significant harmonics (legs + arm swing + torso bob)
            n_harmonics = rng.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
            # Wider range spread: upright posture, ~1.5-1.8m height
            spread = rng.uniform(8, 18)
            # Amplitude per harmonic (fundamental strongest)
            base_amplitude = rng.uniform(0.4, 1.2)
            # Harmonic decay: arms/legs produce stronger higher harmonics
            harmonic_decay = rng.uniform(0.5, 0.8)
            # SNR bias from real data: humans tend to have higher SNR
            snr_db = rng.normal(15, 8)  # mean 15 dB, std 8 dB
        else:
            # Quadruped gait: stride ~1.5-3.0 Hz (faster leg turnover)
            stride_freq = rng.uniform(1.5, 3.0)
            # Animals have 1-3 harmonics (four legs but more regular pattern)
            n_harmonics = rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            # Narrower range spread: lower body profile
            spread = rng.uniform(3, 10)
            # Generally weaker returns (smaller radar cross section)
            base_amplitude = rng.uniform(0.2, 0.8)
            # Faster harmonic decay: more uniform leg motion
            harmonic_decay = rng.uniform(0.3, 0.6)
            # SNR bias from real data: animals tend to have lower SNR
            snr_db = rng.normal(5, 8)  # mean 5 dB, std 8 dB

        # Clamp SNR to physically reasonable range (-5 to 30 dB)
        snr_db = np.clip(snr_db, -5, 30)

        # Range profile (Gaussian around center)
        range_profile = np.exp(-0.5 * ((range_bins - center_bin) / spread) ** 2)

        # Slow-time micro-Doppler signal with harmonics of stride frequency
        doppler_signal = np.zeros(n_slow, dtype=complex)
        for h in range(1, n_harmonics + 1):
            phase = rng.uniform(0, 2 * np.pi)
            amp = base_amplitude * (harmonic_decay ** (h - 1))
            doppler_signal += amp * np.exp(
                1j * (2 * np.pi * stride_freq * h * slow_t + phase)
            )

        # Combine: outer product of range profile and slow-time signal
        target = range_profile[:, None] * doppler_signal[None, :]

        # Convert SNR (dB) to noise level relative to signal
        signal_power = np.mean(np.abs(target) ** 2)
        if signal_power > 0:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_std = np.sqrt(noise_power / 2)  # /2 for real+imag parts
        else:
            noise_std = 0.5

        clutter = noise_std * (
            rng.randn(n_range, n_slow) + 1j * rng.randn(n_range, n_slow)
        )

        segment = clutter + target
        segments.append(segment)

    return np.array(segments)


def generate_synthetic_mafat_data(save_dir, n_samples=3000, seed=42):
    """Generate synthetic radar I/Q data mimicking the real MAFAT format.

    Generates full 128x32 complex I/Q matrices (like real data), then
    processes them through the same I/Q averaging pipeline to produce
    the final (n_samples, 32, 2) output.

    Args:
        save_dir: Directory to save processed data.
        n_samples: Total number of samples (split equally between classes).
        seed: Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    n_human = n_samples // 2
    n_animal = n_samples - n_human

    print("Generating synthetic human radar segments...")
    human_segments = _generate_synthetic_iq_segments(n_human, rng, "human")
    print("Generating synthetic animal radar segments...")
    animal_segments = _generate_synthetic_iq_segments(n_animal, rng, "animal")

    all_segments = np.concatenate([human_segments, animal_segments], axis=0)
    labels = np.concatenate([
        np.ones(n_human, dtype=np.int32),
        np.zeros(n_animal, dtype=np.int32),
    ])

    # Shuffle
    perm = rng.permutation(len(all_segments))
    all_segments = all_segments[perm]
    labels = labels[perm]

    # Process through the same I/Q pipeline as real data:
    # Average across range bins (axis=1 of 128x32) → (32,) complex → (32, 2) real/imag
    mean_signals = np.mean(all_segments, axis=1)  # (N, 32) complex
    data = np.stack([np.real(mean_signals), np.imag(mean_signals)], axis=-1)  # (N, 32, 2)
    data = data.astype(np.float32)

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
    print(f"  Class balance: {dict(zip(*np.unique(np.array(labels), return_counts=True)))}")
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
        # Remove stale processed data if re-processing
        stale = os.path.join(save_dir, "data.pkl")
        if os.path.exists(stale):
            print(f"Note: {stale} already exists. Delete it to re-process.")
        if args.mode == "iq":
            process_mafat_iq(raw_dir, save_dir)
        elif args.mode == "fft":
            process_mafat_fft(raw_dir, save_dir)
