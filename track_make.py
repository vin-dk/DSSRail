import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# === SETUP ===
np.random.seed(42)
n_months = 30  # 15 years, monthly records
n_files = 300   # Number of datasets
output_dir = Path("synthetic_track_dataset")
output_dir.mkdir(exist_ok=True)

# === Degradation Parameters ===
log_std = 0.9
log_mean = 0.08  # Adjusted to reflect a slower monthly increase

# === Maintenance Thresholds ===
PM_THRESHOLD = 1.5
EM_THRESHOLD = 2.2

# Manually specified thresholds
C0 = 1.5   # Class 0 vs 1 cutoff
C1 = 4.5   # Class 1 vs 2 cutoff
beta = 3.0

def assign_defect(dll):
    """
    Uses logistic cutoffs to assign class label (0, 1, or 2).
    DLL_s is the only input feature.
    """
    # Calculate cumulative probabilities
    p_leq_0 = 1 / (1 + np.exp(-(C0 + beta * dll)))
    p_leq_1 = 1 / (1 + np.exp(-(C1 + beta * dll)))

    p0 = p_leq_0
    p1 = p_leq_1 - p_leq_0
    p2 = 1 - p_leq_1

    # Slight fuzziness can be added by sampling
    return np.random.choice([0, 1, 2], p=[p0, p1, p2])

def apply_recovery(value, tamp_type):
    """Reduce DLL_s after tamping event."""
    if tamp_type == 1:  # PM
        base, scale = 0.65, 0.4
    else:  # EM
        base, scale = 0.45, 0.3

    dll_factor = min(1.0, value / 3.0)
    reduction = base + scale * dll_factor
    return max(0.0, value * (1 - reduction))

# === MAIN LOOP ===
all_defects = []

for run in range(n_files):
    start_date = datetime(2000, 1, 1)
    dates = [start_date + timedelta(days=30 * i) for i in range(n_months)]

    dll_s = []
    tamping_flags = []
    tamping_types = []
    defects = []

    current_dll = np.random.uniform(0.4, 0.9)
    last_pm = -365
    last_em = -180

    for i, date in enumerate(dates):
        month = i + 1

        # Degrade
        b_s = np.random.lognormal(mean=log_mean, sigma=log_std)
        b_s = np.clip(b_s, 0, 1.4)  # Cap the degradation increment to a reasonable max
        current_dll += b_s

        tamping_done = 0
        tamping_type = 0

        # Save DLL_s before any recovery
        dll_before_recovery = current_dll

        # Recovery triggers
        if current_dll >= EM_THRESHOLD and (month - last_em) >= 3:
            tamping_done = 1
            tamping_type = 2
            current_dll = apply_recovery(current_dll, tamping_type)
            last_em = month
        elif current_dll >= PM_THRESHOLD and (month - last_pm) >= 6:
            tamping_done = 1
            tamping_type = 1
            current_dll = apply_recovery(current_dll, tamping_type)
            last_pm = month

        defect_class = assign_defect(dll_before_recovery)

        dll_s.append(dll_before_recovery)
        tamping_flags.append(tamping_done)
        tamping_types.append(tamping_type)
        defects.append(defect_class)

    all_defects.extend(defects)

    df = pd.DataFrame({
        "Date": dates,
        "DLL_s": dll_s,
        "Tamping Performed": tamping_flags,
        "Tamping Type": tamping_types,
        "Defect": defects
    })

    file_path = output_dir / f"synthetic_track_data_{run+1:03d}.xlsx"
    df.to_excel(file_path, index=False)

# === Summary of Defects ===
counts = Counter(all_defects)
print("\nDefect Summary Across All Files:")
for defect_class in [0, 1, 2]:
    print(f"  → Class {defect_class}: {counts.get(defect_class, 0)} occurrences")

print(f"[✓] Generated {n_files} synthetic files in {output_dir.resolve()}")