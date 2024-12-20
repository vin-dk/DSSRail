import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

def generate_dates(start_date, total_months):
    """Generate an array of dates, one month apart."""
    dates = [start_date + timedelta(days=30 * i) for i in range(total_months)]
    return dates

def generate_dlls_and_tamping(dates, al=1.5, il=2.0, ial=3.0, initial_dll=0.5):
    """
    Generate DLLs, tamping flags, and tamping types with realistic behavior.
    """
    total_months = len(dates)
    dlls = []
    tamping_performed = []
    tamping_type = []

    current_dll = initial_dll

    # Recovery and degradation parameters
    recovery_noise_std = 0.15
    degradation_rate_shape = 2.379  # Shape of gamma distribution for degradation
    degradation_rate_scale = 0.756  # Scale of gamma distribution for degradation
    catastrophic_failure_prob = 0.05  # 5% chance of catastrophic failure

    for i in range(total_months):
        # Record current DLL
        if current_dll == 0:  # Replace 0 value with a random value between 0.1 and 0.5
            current_dll = random.uniform(0.1, 0.5)

        dlls.append(current_dll)

        # Determine if tamping is needed
        if current_dll > ial:  # Critical defect (emergency CM)
            tamping_performed.append(1)
            tamping_type.append(2)  # Complete tamping
            recovery_noise = np.random.normal(0, np.sqrt(recovery_noise_std))
            current_dll -= 0.269 + 0.51 * current_dll + 0.207 * 2 - 0.043 * 2 * current_dll + recovery_noise
        elif il < current_dll <= ial:  # High defect probability (partial CM)
            tamping_performed.append(1)
            tamping_type.append(1)  # Partial tamping
            recovery_noise = np.random.normal(0, np.sqrt(recovery_noise_std))
            current_dll -= 0.269 + 0.51 * current_dll + 0.207 * 1 - 0.043 * 1 * current_dll + recovery_noise
        elif al < current_dll <= il:  # Preventive maintenance (complete tamping)
            tamping_performed.append(1)
            tamping_type.append(2)  # Complete tamping
            recovery_noise = np.random.normal(0, np.sqrt(recovery_noise_std))
            current_dll -= 0.269 + 0.51 * current_dll + 0.207 * 2 - 0.043 * 2 * current_dll + recovery_noise
        else:  # No tamping
            tamping_performed.append(0)
            tamping_type.append(0)

        # Ensure DLL_s does not go negative
        current_dll = max(current_dll, 0)

        # Apply degradation if no tamping
        if tamping_performed[-1] == 0:
            # Base degradation rate
            degradation_rate = np.random.gamma(degradation_rate_shape, degradation_rate_scale)

            # Adjust degradation rate based on traffic patterns
            if i % 12 < 6:  # Light traffic
                degradation_rate *= 0.5
            else:  # Heavy traffic
                degradation_rate *= 1.5

            # Introduce random catastrophic failures
            if random.random() < catastrophic_failure_prob:
                degradation_rate += random.uniform(0.5, 1.0)

            current_dll += degradation_rate

    return dlls, tamping_performed, tamping_type

def write_to_excel(dates, dlls, tamping_performed, tamping_type, file_path):
    """Write the generated data to an Excel file."""
    data = {
        "Date": [date.strftime('%Y-%m-%d') for date in dates],
        "DLL_s Measurement": dlls,
        "Tamping Performed": tamping_performed,
        "Tamping Type": tamping_type,
    }
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"Data successfully written to {file_path}")

def main():
    start_date = datetime(2002, 1, 1)
    total_months = 200  # Extend the time range for more data
    output_file = "mock_track_data.xlsx"

    dates = generate_dates(start_date, total_months)
    dlls, tamping_performed, tamping_type = generate_dlls_and_tamping(dates)

    write_to_excel(dates, dlls, tamping_performed, tamping_type, output_file)

if __name__ == "__main__":
    main()