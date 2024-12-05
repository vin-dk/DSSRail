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
        start_row = data.loc[tamping_indices[i]]
        end_row = data.loc[tamping_indices[i + 1]]
        delta_DLL = end_row["DLL_s Measurement"] - start_row["DLL_s Measurement"]
        delta_time = (end_row["Date"] - start_row["Date"]).days
        degradation_rate = delta_DLL / delta_time
        degradation_rates.append(degradation_rate)
    results["degradation_rate_mean"] = np.mean(degradation_rates)
    results["degradation_rate_stddev"] = np.std(degradation_rates)

    ## 3. Defect Probability Coefficients
    AL, IL, IAL = 1.5, 2.0, 3.0
    data["Defect Level"] = np.select(
        [data["DLL_s Measurement"] <= AL, 
         (data["DLL_s Measurement"] > AL) & (data["DLL_s Measurement"] <= IL),
         data["DLL_s Measurement"] > IL],
        [1, 2, 3]
    )
    X = data[["DLL_s Measurement"]].values
    y = data["Defect Level"].values
    if len(X) < 3:
        raise ValueError("Insufficient data points for logistic regression.")
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs").fit(X, y)
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


def degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t, noise_stddev=0.1):
    """
    Models the degradation progression over a given time interval.
    
    Parameters:
    - current_DLL_s: float, the current degradation value (DLL_s).
    - degradation_rate_mean: float, the mean degradation rate (b_mean).
    - degradation_rate_stddev: float, the standard deviation of the degradation rate (b_std).
    - delta_t: float, the elapsed time interval (in days).
    - noise_stddev: float, the standard deviation of the random noise (default = 0.1).
    
    Returns:
    - Updated degradation value (DLL_s) for the next interval.
    """
    # Sample the degradation rate from a normal distribution
    b_s = np.random.normal(degradation_rate_mean, degradation_rate_stddev)
    
    # Sample the random noise from a normal distribution
    epsilon_s = np.random.normal(0, noise_stddev)
    
    # Compute the updated degradation value
    next_DLL_s = current_DLL_s + b_s * delta_t + epsilon_s
    
    return max(0, next_DLL_s)  # Ensure degradation value is non-negative


def defect(current_DLL_s, defect_coefficients):
    """
    Calculates defect probabilities for the current degradation state.
    
    Parameters:
    - current_DLL_s: float, the current degradation state (DLL_s).
    - defect_coefficients: dict, containing logistic regression coefficients:
        - "C_0": Coefficient for P1.
        - "C_1": Coefficient for P2.
        - "b": Coefficient for DLL_s.

    Returns:
    - dict containing:
        - "P1": Probability of no defect.
        - "P2": Probability of Defect Level A.
        - "P3": Probability of Defect Level B.
    """
    # Extract coefficients
    C_0 = defect_coefficients["C_0"]
    C_1 = defect_coefficients["C_1"]
    b = defect_coefficients["b"]

    # Logistic regression calculations
    exp_term_0 = np.exp(C_0 + b * current_DLL_s)
    exp_term_1 = np.exp(C_1 + b * current_DLL_s)

    # Calculate probabilities
    P1 = exp_term_0 / (1 + exp_term_0)
    P2 = exp_term_1 / (1 + exp_term_1) - P1
    P3 = 1 - (P1 + P2)

    # Ensure probabilities are valid (sum to 1)
    probabilities = {
        "P1": max(0, min(1, P1)),
        "P2": max(0, min(1, P2)),
        "P3": max(0, min(1, P3)),
    }

    return probabilities


def recovery(current_DLL_s, recovery_coefficients, tamping_type, noise_stddev=0.05):
    """
    Models the recovery of degradation after a tamping event.
    
    Parameters:
    - current_DLL_s: float, the current degradation value (DLL_s) before recovery.
    - recovery_coefficients: dict, containing recovery model coefficients:
        - "alpha_1": Baseline recovery intercept.
        - "beta_1": Effect of DLL_s before tamping.
        - "beta_2": Effect of tamping type.
        - "beta_3": Interaction effect of tamping type and DLL_s.
    - tamping_type: int, the type of tamping (1 = complete, 0 = partial).
    - noise_stddev: float, the standard deviation of the random noise (default = 0.05).
    
    Returns:
    - Updated degradation value (DLL_s) after recovery.
    """
    # Extract coefficients
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]
    
    # Sample the random noise
    epsilon_LL = np.random.normal(0, noise_stddev)
    
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
            defect_coefficients, recovery_coefficients, AL=1.5, IL=2.0, noise_stddev=0.1):
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
        - Defect probabilities (P1, P2, P3).
        - Tamping status (None, partial, or complete).
    """
    # Step 1: Degrade
    next_DLL_s = degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t, noise_stddev)

    # Step 2: Defect
    defect_probs = defect(next_DLL_s, defect_coefficients)

    # Step 3: Recovery (if necessary)
    tamping_status = None
    if next_DLL_s > IL:
        # Complete tamping (PM)
        tamping_status = "complete"
        next_DLL_s = recovery(next_DLL_s, recovery_coefficients, tamping_type=1, noise_stddev=noise_stddev)
    elif AL < next_DLL_s <= IL:
        # Partial tamping (CM)
        tamping_status = "partial"
        next_DLL_s = recovery(next_DLL_s, recovery_coefficients, tamping_type=0, noise_stddev=noise_stddev)

    # Return results for the next step
    return {
        "updated_DLL_s": next_DLL_s,
        "defect_probabilities": defect_probs,
        "tamping_status": tamping_status
    }


def take_in2(previous_results):
    """
    Prepares for the next simulation step by updating variables and recalculating if needed.

    Parameters:
    - previous_results: dict, output from the `compute` method of the previous step.

    Returns:
    - dict containing updated parameters for the next step.
    """
    # Extract data from the previous step
    updated_DLL_s = previous_results["updated_DLL_s"]
    defect_probabilities = previous_results["defect_probabilities"]
    tamping_status = previous_results["tamping_status"]

    # Prepare for the next interval (DLL_s carries forward)
    return {
        "current_DLL_s": updated_DLL_s,
        "defect_probabilities": defect_probabilities,
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
            "current_DLL_s": current_DLL_s,
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