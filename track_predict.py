import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

def take_in(file_path):
    # Load the data
    data = pd.read_excel(file_path)
    
    # Ensure required columns are present
    required_columns = ["Date", "DLL_s Measurement", "Tamping Performed", "Tamping Type"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")
    
    # Convert the Date column to datetime for easier manipulation
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by="Date")
    results = {}

    ## 1. Compute D0LL_s
    most_recent_tamping = data[data["Tamping Performed"] == 1]
    if most_recent_tamping.empty:
        raise ValueError("No tamping data available to compute initial degradation.")
    most_recent_tamping = most_recent_tamping.iloc[-1]
    D0LL_s = most_recent_tamping["DLL_s Measurement"]
    results["D0LL_s"] = D0LL_s

    ## 2. Degradation Rates
    tamping_indices = data[data["Tamping Performed"] == 1].index.tolist()
    if len(tamping_indices) < 2:
        raise ValueError("Insufficient tamping data to calculate degradation rates.")
    
    degradation_rates = []
    for i in range(len(tamping_indices) - 1):
        # Skip the row of the tamping event to avoid including recovery effects
        start_row = data.loc[tamping_indices[i] + 1] if (tamping_indices[i] + 1 < len(data)) else None
        end_row = data.loc[tamping_indices[i + 1]]
        
        if start_row is not None:
            delta_DLL = end_row["DLL_s Measurement"] - start_row["DLL_s Measurement"]
            delta_time = (end_row["Date"] - start_row["Date"]).days
            if delta_time > 0:  # Avoid zero or negative intervals
                degradation_rate = delta_DLL / delta_time
                degradation_rates.append(degradation_rate)
    
    if not degradation_rates:
        raise ValueError("No valid degradation rates found.")
    
    results["degradation_rate_mean"] = np.mean(degradation_rates)
    results["degradation_rate_stddev"] = np.std(degradation_rates)

    ## 3. Defect Probability Coefficients
    AL, IL, IAL = 1.0, 1.5, 2.0  # Limits as defined in the paper
    data["Defect Level"] = np.select(
        [data["DLL_s Measurement"] <= AL, 
         (data["DLL_s Measurement"] > AL) & (data["DLL_s Measurement"] <= IL),
         data["DLL_s Measurement"] > IL],
        [1, 2, 3]
    )
    X = data[["DLL_s Measurement"]].values
    y = data["Defect Level"].values
    print(f"Unique classes in y: {np.unique(y)}")
    print(data["Defect Level"].value_counts())
    if len(X) < 3:
        raise ValueError("Insufficient data points for logistic regression.")
    model = LogisticRegression(solver="lbfgs", multi_class="multinomial").fit(X, y)
    results["defect_coefficients"] = {
        "C_0": model.intercept_[0],
        "C_1": model.intercept_[1],
        "b": model.coef_[0][0]
    }

    ## 4. Recovery Coefficients
    recovery_data = []
    for idx in tamping_indices:
        if idx > 0:
            pre_tamping = data.loc[idx - 1]
            post_tamping = data.loc[idx]
            R_LL_s = pre_tamping["DLL_s Measurement"] - post_tamping["DLL_s Measurement"]
            recovery_data.append({
                "DLL_s": pre_tamping["DLL_s Measurement"],
                "x_s": post_tamping["Tamping Type"],
                "R_LL_s": R_LL_s
            })
    if len(recovery_data) < 2:
        raise ValueError("Insufficient recovery data for regression.")
    recovery_df = pd.DataFrame(recovery_data)
    X = recovery_df[["DLL_s", "x_s"]]
    X["Interaction"] = recovery_df["DLL_s"] * recovery_df["x_s"]
    y = recovery_df["R_LL_s"]
    model = LinearRegression().fit(X, y)
    results["recovery_coefficients"] = {
        "alpha_1": model.intercept_,
        "beta_1": model.coef_[0],
        "beta_2": model.coef_[1],
        "beta_3": model.coef_[2],
    }

    return results


def degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t):
    """
    Models the degradation progression over a given time interval.
    """
    # Sample the degradation rate from a normal distribution
    b_s = np.random.normal(degradation_rate_mean, degradation_rate_stddev)
    
    # Sample the random noise from a normal distribution using degradation rate stddev
    epsilon_s = np.random.normal(0, degradation_rate_stddev)  
    
    # Compute the updated degradation value
    next_DLL_s = current_DLL_s + b_s * delta_t + epsilon_s
    
    return max(0, next_DLL_s)  # Ensure degradation value is non-negative

def defect(current_DLL_s, defect_coefficients):
    """
    Calculates defect probabilities for the current degradation state using a softmax approach.
    """
    # Extract coefficients
    C_0 = defect_coefficients["C_0"]
    C_1 = defect_coefficients["C_1"]
    b = defect_coefficients["b"]

    # Calculate logits for each class
    logits = np.array([
        C_0 + b * current_DLL_s,  # Logit for Defect Level 1 (P1)
        C_1 + b * current_DLL_s,  # Logit for Defect Level 2 (P2)
        0                         # Logit for Defect Level 3 (baseline class)
    ])

    # Apply the softmax function
    exp_logits = np.exp(logits - np.max(logits))  # Stability adjustment
    probabilities = exp_logits / np.sum(exp_logits)

    # Map to named probabilities
    probabilities_dict = {
        "P1": probabilities[0],
        "P2": probabilities[1],
        "P3": probabilities[2],
    }

    return probabilities_dict


def recovery(current_DLL_s, recovery_coefficients, tamping_type):
    """
    Models the recovery of degradation after a tamping event.
    """
    # Extract coefficients
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]
    
    noise_stddev=np.sqrt(0.15)
    
    # Sample the random noise
    epsilon_LL = np.random.normal(0, noise_stddev)  # Fixed noise stddev as per the paper
    
    # Calculate the recovery value (R_LL_s)
    R_LL_s = (
        alpha_1 +
        beta_1 * current_DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * current_DLL_s +
        epsilon_LL
    )
    
    # Update the degradation value after recovery
    updated_DLL_s = max(0, current_DLL_s - R_LL_s)  # Ensure non-negative DLL_s
    
    return updated_DLL_s

def compute(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t, 
            defect_coefficients, recovery_coefficients, AL=1.0, IL=1.5, noise_stddev=0.1):
    """
    Executes one step of the simulation process.

    Parameters:
    - current_DLL_s: float, the current degradation value (DLL_s).
    - degradation_rate_mean: float, mean degradation rate (b_mean).
    - degradation_rate_stddev: float, standard deviation of degradation rate (b_std).
    - delta_t: float, time interval (in days).
    - defect_coefficients: dict, logistic regression coefficients for defect probabilities.
    - recovery_coefficients: dict, recovery model coefficients.
    - AL: float, Alert Limit.
    - IL: float, Intervention Limit.
    - noise_stddev: float, standard deviation of random noise.

    Returns:
    - dict containing:
        - Updated DLL_s.
        - Degradation value.
        - Recovery adjustment (if recovery is applied).
        - Defect probabilities (P1, P2, P3).
        - Tamping status (None, partial, or complete).
    """
    # Step 1: Degrade
    next_DLL_s_before_recovery = degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t)
    degradation_val = next_DLL_s_before_recovery - current_DLL_s  # Calculate degradation value

    # Step 2: Defect
    defect_probs = defect(next_DLL_s_before_recovery, defect_coefficients)

    # Step 3: Recovery (if necessary)
    tamping_status = None
    recovery_adjustment = 0  # Default is no recovery adjustment
    updated_DLL_s = next_DLL_s_before_recovery  # Default case: no tamping/recovery
    if next_DLL_s_before_recovery > IL:
        # Partial tamping (CM, corrective)
        tamping_status = "partial"
        updated_DLL_s = recovery(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=0)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s
    elif AL < next_DLL_s_before_recovery <= IL:
        # Complete tamping (PM, preventive)
        tamping_status = "complete"
        updated_DLL_s = recovery(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=1)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s

    # Return results for the next step
    return {
        "current_DLL_s": current_DLL_s,
        "degradation_val": degradation_val,
        "recovery_adjustment": recovery_adjustment,  # Log the recovery adjustment
        "updated_DLL_s": updated_DLL_s,
        "defect_probabilities": defect_probs,
        "tamping_status": tamping_status
    }

def sim(file_path, interval_months, time_length_months):
    """
    Simulates the degradation and recovery process for the specified time length
    with a given interval.

    Parameters:
    - file_path: str, path to the input Excel file.
    - interval_months: int, the time interval in months between each step.
    - time_length_months: int, the total simulation time length in months.

    Returns:
    - mass_data: list, containing data for all runs and intervals.
    """
    # Calculate the total number of intervals
    total_intervals = time_length_months // interval_months
    delta_t_days = interval_months * 30  # Approximate days in one month

    # Prepare structure to store all runs
    mass_data = []

    # Load initial data
    input_data = take_in(file_path)
    current_DLL_s = input_data["D0LL_s"]
    degradation_rate_mean = input_data["degradation_rate_mean"]
    degradation_rate_stddev = input_data["degradation_rate_stddev"]
    defect_coefficients = input_data["defect_coefficients"]
    recovery_coefficients = input_data["recovery_coefficients"]

    # Simulate through each interval
    for interval in range(total_intervals):
        print(f"Run {interval + 1}")  # Indicate the current run in the console

        # Perform one simulation step
        step_results = compute(
            current_DLL_s,
            degradation_rate_mean,
            degradation_rate_stddev,
            delta_t_days,
            defect_coefficients,
            recovery_coefficients
        )

        # Collect data for this interval
        interval_data = {
            "run": interval + 1,
            "current_DLL_s": step_results["current_DLL_s"],
            "degradation_val": step_results["degradation_val"],
            "recovery_adjustment": step_results["recovery_adjustment"],  # Show recovery adjustment
            "updated_DLL_s": step_results["updated_DLL_s"],
            "defect_probabilities": step_results["defect_probabilities"],
            "tamping_status": step_results["tamping_status"],
            "degradation_rate_mean": degradation_rate_mean,
            "degradation_rate_stddev": degradation_rate_stddev,
            "recovery_coefficients": recovery_coefficients
        }

        # Append interval data to mass data structure
        mass_data.append(interval_data)

        # Update current DLL_s for the next interval
        current_DLL_s = step_results["updated_DLL_s"]

        # Print interval data to console
        for key, value in interval_data.items():
            print(f"{key}: {value}")
        print("\n")  # Separate runs with a new line

    return mass_data

sim(r"C:\Users\13046\mock_track_data.xlsx", 1, 100)