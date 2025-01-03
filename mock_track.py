import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def gen_mock_data(output_file="mock_data_file.xlsx"):
    """
    Generates a mock dataset for track degradation and maintenance.

    Parameters:
        output_file (str): The name of the Excel file to save the mock dataset.

    Returns:
        None
    """
    # Constants and parameters
    D_0LLs = 0.352  # Initial degradation value after tamping
    degradation_mean = -2.379  # Log mean for degradation rate
    degradation_stddev = 0.756  # Log standard deviation for degradation rate
    e_s_mean = 0  # Mean of the Gaussian error term
    e_s_variance = 0.041  # Variance of the Gaussian error term

    # Recovery model parameters
    recovery_coefficients = {
        "alpha_1": -0.269,
        "beta_1": 0.51,
        "beta_2": 0.207,
        "beta_3": -0.043
    }

    # Simulation settings
    num_records = 200
    start_date = datetime(2000, 1, 1)

    # Data containers
    dates = []
    dll_values = []
    tamping_performed = []
    tamping_type = []

    # Initialize variables
    current_dll = D_0LLs
    current_date = start_date
    degradation_rate = np.random.lognormal(mean=degradation_mean, sigma=degradation_stddev)
    time_since_last_tamping = 0

    for i in range(num_records):
        # Append the current date
        dates.append(current_date)

        # Sample error term
        e_s = np.random.normal(loc=e_s_mean, scale=np.sqrt(e_s_variance))

        # Update DLL_s using the degradation formula
        degraded_dll = current_dll + degradation_rate * time_since_last_tamping + e_s

        # Append the degraded DLL_s value temporarily for evaluation
        dll_values.append(degraded_dll)

        # Check if tamping is required
        if degraded_dll > 1.5:  # AL (Alert Limit)
            tamping_action = 2  # Complete tamping
            tamping_performed.append(1)  # Tamping performed
            tamping_type.append(tamping_action)

            # Recovery calculation
            x_s = 1 if tamping_action == 2 else 0
            epsilon_LL = np.random.normal(0, np.sqrt(0.15))
            R_LL_s = (
                recovery_coefficients["alpha_1"] +
                recovery_coefficients["beta_1"] * degraded_dll +
                recovery_coefficients["beta_2"] * x_s +
                recovery_coefficients["beta_3"] * x_s * degraded_dll +
                epsilon_LL
            )
            current_dll = degraded_dll - R_LL_s  # Apply recovery

            # Reset time since last tamping
            time_since_last_tamping = 0
        else:
            # No tamping performed
            tamping_performed.append(0)
            tamping_type.append(0)
            current_dll = degraded_dll
            time_since_last_tamping += np.random.randint(20, 41)  # Randomize observation interval between 20 and 40 days

        # Ensure DLL_s does not go negative
        current_dll = max(current_dll, 0)

        # Update date based on the randomized observation interval
        current_date += timedelta(days=time_since_last_tamping)

    # Create a DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "DLL_s": dll_values,
        "Tamping Performed": tamping_performed,
        "Tamping Type": tamping_type
    })

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Mock data saved to {output_file}")

# Generate the mock dataset
gen_mock_data()
