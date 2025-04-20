# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import anderson
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# NOTE TO SELF: 
# We need to change the input to the defect model such that each dlls is assigned a 0 - 2 number indicating isolated defect. This helps the model predict isolated defect based on DLLs
# We also need to add an option for the case that we see before and after recovery. That is, we see recovery effect isolated. This helps the recovery model, but IS NOT required. 
# Some other changes, such as to output, may need to be implemented, but for the most part is ok. We at least need base excel output, further analysis done by operator. 

# Also to add, is the defect model, such that we expect 5 columns, with isolated defect indicated

import subprocess
import sys

try:
    import xlsxwriter
except ImportError:
    print("[!] xlsxwriter not found. Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
    import xlsxwriter
    print("[✓] xlsxwriter installed successfully.")


"""
THESE ARE THE PARAMETERS FOR A GOOD RUN
"""
# Constants and parameters
D_0LLs = 0.352  # Initial degradation value after tamping (given in the paper)
degradation_mean = -2.379  # Log mean for degradation rate
degradation_stddev = 0.756  # Log standard deviation for degradation rate
e_s_mean = 0  # Mean of the Gaussian error term
e_s_variance = np.sqrt(0.041)  # Variance of the Gaussian error term (converted to standard dev for simplicity in later calcs)

# Parameters for recovery model
recovery_coefficients = {
    "alpha_1": -0.269,
    "beta_1": 0.51,
    "beta_2": 0.207,
    "beta_3": -0.043
}
e_LL_mean = 0  # Mean of the Gaussian noise for recovery
e_LL_variance = np.sqrt( 0.15)  # Variance of the Gaussian noise for recovery (converted to standard dev for simplicity in later calcs)

# Parameters for defect probability model
defect_coefficients = {
    "C0": 9.1875,  # Coefficient for IL level
    "C1": 13.39,   # Coefficient for IAL level
    "b": -4.7712    # Coefficient for degradation predictor
}

# Maintenance cost parameters
inspection_cost = 240  # Cost per inspection (in SEK)
preventive_maintenance_cost = 5000  # Cost of preventive maintenance (SEK)
normal_corrective_maintenance_cost = 11000  # Normal corrective maintenance cost (SEK)
emergency_corrective_maintenance_cost = 40000  # Emergency corrective maintenance cost (SEK)

# Maintenance limit thresholds
AL = 1.5  # Alert limit
IL = 2.0  # Intervention limit
IAL = 3.0  # Immediate action limit

# Monte Carlo simulation parameters
num_simulations = 100  # Total number of simulation runs
inspection_interval = 120  # Time interval between inspections (days)
simulation_time_horizon = 15 * 365  # Total simulation time horizon (15 years)

# Tamping parameters
response_time_mean = 5 * 7  # Mean response time in days (5 weeks)
response_time_stddev = 7  # Standard deviation for response time (1 week)

# Derived constants
time_step = 1  # Daily degradation update

ILL = 0.4    # Trigger normal corrective if P2 > 0.4
IALL = 0.05  # Trigger emergency corrective if P3 > 0.05

"""
PARAMETERS END
"""

def take_in(file_path):
    """
    Reads an Excel file and calculates constants and parameters related to degradation and recovery.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Contains degradation stats, degradation rate list, and post-tamping DLL_s.
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import anderson

    # === Input Validation ===
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string pointing to an Excel file.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df.empty:
        raise ValueError("Input file is empty.")

    # Normalize DLL_s column
    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Input file must contain 'DLL_s', 'A', or 'G' column.")

    # Ensure all required columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    # Validate and convert dates
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        raise ValueError("Failed to parse 'Date' column as datetime.")

    # Sort chronologically
    df = df.sort_values(by="Date")

    # Time interval (in days)
    df["Time Interval"] = (df["Date"] - df["Date"].shift(1)).dt.days

    # Identify tamping events
    tamping_indices = df[df["Tamping Performed"] == 1].index.tolist()
    if len(tamping_indices) < 2:
        raise ValueError("At least two tamping events are required to calculate degradation rates.")

    # Slice into degradation periods
    degradation_periods = []
    for i in range(len(tamping_indices) - 1):
        start = tamping_indices[i] + 1
        end = tamping_indices[i + 1]
        degradation_periods.append(df.loc[start:end])

    degradation_rates = []
    time_intervals = []
    for period in degradation_periods:
        period["Degradation Rate"] = period["DLL_s"].diff() / period["Time Interval"]
        degradation_rates.extend(period["Degradation Rate"].dropna().tolist())
        time_intervals.extend(period["Time Interval"].dropna().tolist())

    # Lognormality check
    dll_s_positive = df["DLL_s"][df["DLL_s"] > 0]
    if dll_s_positive.empty:
        raise ValueError("DLL_s must contain at least one positive value for lognormality check.")

    log_dll_values = np.log(dll_s_positive)
    result = anderson(log_dll_values, dist="norm")

    if result.statistic < result.critical_values[2]:  # 5% significance level
        print("DLL_s values follow a lognormal distribution.")
        lognormal_mean = np.mean(log_dll_values)
        lognormal_stddev = np.std(log_dll_values)
        degradation_mean = np.exp(lognormal_mean + 0.5 * lognormal_stddev**2)
        degradation_stddev = np.sqrt(
            (np.exp(lognormal_stddev**2) - 1) * np.exp(2 * lognormal_mean + lognormal_stddev**2)
        )
    else:
        print("DLL_s values do not follow a lognormal distribution. Assuming normal distribution.")
        degradation_mean = np.mean(df["DLL_s"])
        degradation_stddev = np.std(df["DLL_s"])

    print(f"Degradation Mean: {degradation_mean}")
    print(f"Degradation Stddev: {degradation_stddev}")

    # Get D_0LLs: post-last tamping value
    most_recent_tamping = df[df["Tamping Performed"] == 1].iloc[-1]
    most_recent_index = most_recent_tamping.name
    if most_recent_index + 1 < len(df):
        D_0LLs = df.iloc[most_recent_index + 1]["DLL_s"]
    else:
        D_0LLs = most_recent_tamping["DLL_s"]

    print(f"D_0LLs (Initial degradation value after tamping): {D_0LLs}")

    return {
        "D_0LLs": D_0LLs,
        "degradation_mean": degradation_mean,
        "degradation_stddev": degradation_stddev,
        "degradation_rates": degradation_rates,
        "time_intervals": time_intervals,
    }


def derive_recovery(file_path):
    """
    Derives recovery coefficients from historical track data.
    This version does NOT adjust for natural degradation (use derive_accurate_recovery() for that).

    Returns:
        dict: Recovery model coefficients and residual stats (mean/std).
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    # === Input Checks ===
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string path to an Excel file.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    if df.empty:
        raise ValueError("The input Excel file is empty.")

    # Normalize DLL_s column
    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Missing DLL_s column. Expected one of: DLL_s, A, G.")

    # Required columns
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    if df["DLL_s"].std() < 1e-6:
        raise ValueError("DLL_s column has near-zero variance. Regression will be unstable.")

    if (df["DLL_s"] < 0).any():
        raise ValueError("DLL_s contains negative values, which is not physically valid.")

    # Filter tamping events
    tamping_events = df[df["Tamping Performed"] == 1]
    if len(tamping_events) < 4:
        raise ValueError("Not enough tamping events (min 4 required) for regression.")

    # Recovery effect: DLL_s(before) - DLL_s(after)
    DLL_s_values = df["DLL_s"].values
    R_LL_s = []
    recovery_indices = []

    for idx in tamping_events.index:
        if idx + 1 < len(df):
            recovery = DLL_s_values[idx] - DLL_s_values[idx + 1]
            R_LL_s.append(recovery)
            recovery_indices.append(idx)

    if len(R_LL_s) < 3:
        raise ValueError("Not enough valid recovery pairs for regression (min 3).")

    # Build predictors
    tamping_events = df.loc[recovery_indices]
    X = tamping_events[["DLL_s", "Tamping Type"]].copy()
    X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})
    X["Interaction"] = X["DLL_s"] * X["Tamping Type"]

    Y = np.array(R_LL_s)
    X = X.values

    # Linear regression
    model = LinearRegression()
    model.fit(X, Y)

    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    print(f"Derived Recovery Coefficients:")
    print(f"α₁ (Intercept): {alpha_1}")
    print(f"β₁ (DLL_s): {beta_1}")
    print(f"β₂ (Tamping Type): {beta_2}")
    print(f"β₃ (Interaction): {beta_3}")

    # Residual analysis
    residuals = Y - model.predict(X)
    ad_test_result = anderson(residuals, dist="norm")

    if ad_test_result.statistic < ad_test_result.critical_values[2]:
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        print("Residuals follow normal distribution.")
    else:
        print("Residuals DO NOT follow a normal distribution. Simulation will still proceed.")
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)

    return {
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "e_s_mean": residual_mean,
        "e_s_stddev": residual_stddev,
    }


# This is the true method 
def derive_accurate_recovery(file_path, bs_values):
    """
    Derives adjusted recovery coefficients accounting for degradation during the tamping-to-next interval.
    Adjusted recovery: R_LL_s + (average_b_s * delta_time)

    Returns:
        dict: Recovery coefficients and residual stats.
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    # === Input Validation ===
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string to an Excel file.")

    if not hasattr(bs_values, "__iter__") or len(bs_values) == 0:
        raise ValueError("bs_values must be a non-empty iterable of numeric degradation rates.")

    bs_values = np.asarray(bs_values)
    if not np.isfinite(bs_values).all():
        raise ValueError("bs_values contains NaNs or non-numeric values.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df.empty:
        raise ValueError("Input file is empty.")

    # Normalize DLL_s column
    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Missing DLL_s. Expected one of 'DLL_s', 'A', or 'G'.")

    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    # Ensure valid dates
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        raise ValueError("Failed to convert 'Date' to datetime format.")

    # Validate DLL_s domain
    if (df["DLL_s"] < 0).any():
        raise ValueError("DLL_s contains negative values.")

    if df["DLL_s"].std() < 1e-6:
        raise ValueError("DLL_s has too little variance to train a meaningful model.")

    # Filter tamping rows
    tamping_events = df[df["Tamping Performed"] == 1]
    if len(tamping_events) < 4:
        raise ValueError("Not enough tamping events (min 4 required).")

    average_bs = np.mean(bs_values)

    # === Recovery Calculation ===
    DLL_s_values = df["DLL_s"].values
    R_LL_s_adjusted = []
    valid_indices = []

    tamping_indices = tamping_events.index.tolist()

    for i in range(len(tamping_indices) - 1):
        idx = tamping_indices[i]
        next_idx = tamping_indices[i + 1]

        if next_idx >= len(df):
            continue

        t1 = df.loc[idx, "Date"]
        t2 = df.loc[next_idx, "Date"]
        interval = (t2 - t1).days

        if interval <= 0:
            continue

        observed_R = DLL_s_values[idx] - DLL_s_values[next_idx]
        adjusted_R = observed_R + (average_bs * interval)

        R_LL_s_adjusted.append(adjusted_R)
        valid_indices.append(idx)

    if len(R_LL_s_adjusted) < 3:
        raise ValueError("Not enough valid recovery intervals (need ≥ 3).")

    tamping_events = df.loc[valid_indices]
    X = tamping_events[["DLL_s", "Tamping Type"]].copy()
    X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})
    X["Interaction"] = X["DLL_s"] * X["Tamping Type"]

    X = X.values
    Y = np.array(R_LL_s_adjusted)

    model = LinearRegression()
    model.fit(X, Y)

    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    print("Derived Accurate Recovery Coefficients:")
    print(f"α₁ (Intercept): {alpha_1}")
    print(f"β₁ (DLL_s): {beta_1}")
    print(f"β₂ (Tamping Type): {beta_2}")
    print(f"β₃ (Interaction): {beta_3}")

    residuals = Y - model.predict(X)
    ad_test_result = anderson(residuals, dist="norm")

    if ad_test_result.statistic < ad_test_result.critical_values[2]:
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        print("Residuals follow normal distribution.")
    else:
        print("Residuals do NOT follow normal distribution. Using estimated mean/std anyway.")
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)

    return {
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "e_s_mean": residual_mean,
        "e_s_stddev": residual_stddev,
    }

# Example 
# coefficients = derive_recovery("mock_data_file.xlsx")
# print(coefficients)

# We need alt recovery method here which expects a recovery JUST BEFORE, and JUST AFTER tamping


def derive_defect_probabilities(file_path):
    """
    Derives defect classification parameters using ordinal logistic regression.

    Returns:
        dict: Thresholds (C0, C1), beta, and model pseudo R^2.
    """
    import pandas as pd
    import numpy as np
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    # === Input Validation ===
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a valid Excel file path.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df.empty:
        raise ValueError("Input data file is empty.")

    # DLL_s column remapping
    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Missing DLL_s column. Expected one of: DLL_s, A, G.")

    # Required columns
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    if df["DLL_s"].std() < 1e-6:
        raise ValueError("DLL_s variance too low. Model will be unstable.")

    if (df["DLL_s"] < 0).any():
        raise ValueError("DLL_s contains negative values, which is invalid.")

    # Check for valid defect values
    if "Defect" not in df.columns:
        raise ValueError("Missing 'Defect' column.")

    if not set(df["Defect"].unique()).issubset({0, 1, 2}):
        raise ValueError("Defect values must be only 0, 1, or 2.")

    # Ensure all classes are represented
    class_counts = df["Defect"].value_counts()
    if class_counts.min() < 2:
        raise ValueError("One or more defect classes are underrepresented (< 2 samples).")

    # === Ordinal Model Fit ===
    X = df[["DLL_s"]]
    Y = df["Defect"]

    try:
        model = OrderedModel(Y, X, distr="logit")
        results = model.fit(method="bfgs", disp=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fit ordinal model: {e}")

    if not results.mle_retvals.get("converged", False):
        raise RuntimeError("Ordinal model did not converge.")

    params = results.params

    threshold_keys = [k for k in params.index if "threshold" in k]
    if len(threshold_keys) != 2:
        raise ValueError("Expected exactly two thresholds (C0, C1) for 3-class model.")

    thresholds = {k: params[k] for k in sorted(threshold_keys)}
    C0 = thresholds[sorted(threshold_keys)[0]]
    C1 = thresholds[sorted(threshold_keys)[1]]

    if C0 >= C1:
        raise ValueError(f"Thresholds not in expected order: C0={C0}, C1={C1}")

    if "DLL_s" not in params:
        raise ValueError("Model did not estimate DLL_s coefficient.")

    beta = params["DLL_s"]
    prsquared = results.prsquared

    print("Derived Defect Probability Parameters:")
    print(f"C₀ (0 vs 1): {C0}")
    print(f"C₁ (1 vs 2): {C1}")
    print(f"β (DLL_s): {beta}")
    print(f"Pseudo R²: {prsquared}")

    return {
        "C0": C0,
        "C1": C1,
        "beta": beta,
        "goodness_of_fit": prsquared
    }

# Example 
# coefficients = derive_defect_probabilities("mock_data_file.xlsx")
# print(coefficients)

def degrade(current_DLL_s, b_s, time, e_s):
    """
    Models the degradation progression over a time interval.

    Parameters:
        current_DLL_s (float): Current DLL_s value (≥ 0).
        b_s (float): Degradation rate (> 0).
        time (int): Time step (≥ 0).
        e_s (float): Gaussian error term.

    Returns:
        float: Updated DLL_s value.
    """
    if current_DLL_s < 0:
        raise ValueError("current_DLL_s must be ≥ 0.")

    if b_s <= 0 or not np.isfinite(b_s):
        raise ValueError("b_s must be a positive finite number.")

    if not isinstance(time, (int, float)) or time < 0:
        raise ValueError("time must be a non-negative number.")

    if not np.isfinite(e_s):
        raise ValueError("e_s must be a finite number.")

    return current_DLL_s + (b_s * time) + e_s
    

def recovery(current_DLL_s, recovery_coefficients, tamping_type, re_s_m, re_s_s):
    """
    Applies the recovery model to update DLL_s after tamping.

    Parameters:
        current_DLL_s (float): DLL_s before recovery (≥ 0).
        recovery_coefficients (dict): Must include alpha_1, beta_1, beta_2, beta_3.
        tamping_type (int): 0 for partial, 1 for complete.
        re_s_m (float): Mean of residuals.
        re_s_s (float): Stddev of residuals.

    Returns:
        float: Updated DLL_s after recovery.
    """
    if current_DLL_s < 0:
        raise ValueError("current_DLL_s must be ≥ 0.")

    if tamping_type not in {0, 1}:
        raise ValueError("tamping_type must be 0 (partial) or 1 (complete).")

    expected_keys = {"alpha_1", "beta_1", "beta_2", "beta_3"}
    if not expected_keys.issubset(set(recovery_coefficients)):
        raise ValueError(f"recovery_coefficients must contain: {expected_keys}")

    if re_s_s < 0:
        raise ValueError("re_s_s must be non-negative.")

    epsilon_LL = np.random.normal(re_s_m, re_s_s)

    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]

    R_LL_s = (
        alpha_1 +
        beta_1 * current_DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * current_DLL_s +
        epsilon_LL
    )

    updated = current_DLL_s - R_LL_s
    return max(updated, 0)  # DLL_s can't go negative

def defect(current_DLL_s, defect_coefficients):
    """
    Calculates defect probabilities based on DLL_s using the logistic ordinal model.

    Parameters:
        current_DLL_s (float): DLL_s at inspection.
        defect_coefficients (dict): Must include C0, C1, and beta.

    Returns:
        dict: Probabilities P1 (no defect), P2 (IL), P3 (IAL).
    """
    if current_DLL_s < 0:
        raise ValueError("current_DLL_s must be ≥ 0.")

    required_keys = {"C0", "C1", "beta"}
    if not required_keys.issubset(set(defect_coefficients)):
        raise ValueError(f"defect_coefficients must contain: {required_keys}")

    C0 = defect_coefficients["C0"]
    C1 = defect_coefficients["C1"]
    b = defect_coefficients["beta"]

    if not all(np.isfinite(val) for val in [C0, C1, b]):
        raise ValueError("All coefficients must be finite numbers.")

    P_leq_1 = np.exp(C0 + b * current_DLL_s) / (1 + np.exp(C0 + b * current_DLL_s))
    P_leq_2 = np.exp(C1 + b * current_DLL_s) / (1 + np.exp(C1 + b * current_DLL_s))

    P1 = P_leq_1
    P2 = P_leq_2 - P_leq_1
    P3 = 1 - P1 - P2

    if any(p < 0 or p > 1 for p in [P1, P2, P3]) or not np.isclose(P1 + P2 + P3, 1, atol=1e-4):
        raise ValueError("Calculated probabilities are not valid.")

    return {"P1": P1, "P2": P2, "P3": P3}



def sim_seg(time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, lognormal_mean, lognormal_stddev,
            e_s_mean, e_s_s, re_s_m, re_s_s, ILL, IALL, RM, RV,
            recovery_coefficients, defect_coefficients):
    """
    Simulates a single segment over the defined time horizon.

    Returns:
        dict: Final state and counters.
    """
    if T_insp % T_step != 0 or T_tamp % T_step != 0:
        raise ValueError("T_insp and T_tamp must be multiples of T_step for aligned intervals.")

    t = 1
    tn = 0
    Npm = Ncm_n = Ncm_e = Ninsp = total_response_delay = 0

    b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)
    b_s = min(b_s, 5)  # Cap extremely unrealistic b_s

    while t <= time_horizon:
        t += T_step
        e_s = np.random.normal(e_s_mean, e_s_s)

        DLL_s_t = max(D_0LLs + b_s * (t - tn) + e_s, 0)

        pm_done_today = False

        # Preventive Maintenance
        if t % T_tamp == 0:
            if DLL_s_t > AL:
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 1, re_s_m, re_s_s)
                Npm += 1
                tn = t
                b_s = min(np.random.lognormal(lognormal_mean, lognormal_stddev), 5)
                pm_done_today = True

        # Corrective Maintenance (on inspection day)
        if t % T_insp == 0:
            Ninsp += 1
            if not pm_done_today:
                probabilities = defect(DLL_s_t, defect_coefficients)
                PIL = probabilities["P2"]
                PIAL = probabilities["P3"]

                if PIAL > IALL:
                    D_0LLs = recovery(DLL_s_t, recovery_coefficients, 0, re_s_m, re_s_s)
                    Ncm_e += 1
                    tn = t
                    continue

                elif PIL > ILL:
                    time_to_next_inspection = T_insp - (t % T_insp)
                    response_time = min(max(0, np.random.normal(loc=RM, scale=RV)), time_to_next_inspection)

                    response_time = int(response_time)
                    total_response_delay += response_time
                    t += response_time

                    delayed_e_s = np.random.normal(e_s_mean, e_s_s)
                    DLL_s_t = degrade(DLL_s_t, b_s, response_time, delayed_e_s)

                    D_0LLs = recovery(DLL_s_t, recovery_coefficients, 0, re_s_m, re_s_s)
                    Ncm_n += 1
                    tn = t
                    continue

    return {
        "final_t": t,
        "final_DLL_s": DLL_s_t,
        "b_s": b_s,
        "Npm": Npm,
        "Ncm_n": Ncm_n,
        "Ncm_e": Ncm_e,
        "Total Response Delay": total_response_delay,
        "Number of inspections": Ninsp
    }


def monte(
    time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs,
    degradation_mean, degradation_stddev, e_s_mean, e_s_variance,
    re_s_m, re_s_s, ILL, IALL, RM, RV, num_simulations,
    inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost,
    recovery_coefficients, defect_coefficients
):
    """
    Monte Carlo simulation to aggregate results for multiple track sections with full validation.

    Raises:
        ValueError: On any input that violates expected logic or sanity checks.
    """

    # === Input Validation ===
    if not isinstance(num_simulations, int) or num_simulations < 1:
        raise ValueError("num_simulations must be a positive integer.")

    if not isinstance(time_horizon, int) or time_horizon < 1:
        raise ValueError("time_horizon must be a positive integer.")

    for name, val in [("T_insp", T_insp), ("T_tamp", T_tamp), ("T_step", T_step)]:
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if degradation_stddev <= 0:
        raise ValueError("degradation_stddev must be > 0.")

    if e_s_variance < 0:
        raise ValueError("e_s_variance (variance of degradation error) must be >= 0.")

    if re_s_s < 0:
        raise ValueError("re_s_s (stddev of recovery residuals) must be >= 0.")

    for name, val in [("ILL", ILL), ("IALL", IALL)]:
        if not (0 <= val <= 1):
            raise ValueError(f"{name} must be between 0 and 1 (inclusive).")

    if RM < 0 or RV < 0:
        raise ValueError("RM and RV (response mean/variance) must be non-negative.")

    for name, val in [
        ("inspection_cost", inspection_cost),
        ("preventive_maintenance_cost", preventive_maintenance_cost),
        ("normal_corrective_maintenance_cost", normal_corrective_maintenance_cost),
        ("emergency_corrective_maintenance_cost", emergency_corrective_maintenance_cost),
    ]:
        if val < 0:
            raise ValueError(f"{name} cannot be negative.")

    if D_0LLs < 0:
        raise ValueError("D_0LLs must be non-negative (initial DLL_s after tamping).")

    # === Monte Carlo Simulation ===
    total_inspections = 0
    total_pm = 0
    total_cm_n = 0
    total_cm_e = 0
    total_cost = 0

    for i in range(num_simulations):
        result = sim_seg(
            time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean,
            degradation_stddev, e_s_mean, e_s_variance, re_s_m, re_s_s,
            ILL, IALL, RM, RV,
            recovery_coefficients, defect_coefficients
        )

        inspections = result["Number of inspections"]
        pm = result["Npm"]
        cm_n = result["Ncm_n"]
        cm_e = result["Ncm_e"]

        cost = (
            inspections * inspection_cost +
            pm * preventive_maintenance_cost +
            cm_n * normal_corrective_maintenance_cost +
            cm_e * emergency_corrective_maintenance_cost
        )

        total_inspections += inspections
        total_pm += pm
        total_cm_n += cm_n
        total_cm_e += cm_e
        total_cost += cost

    avg_inspections = total_inspections / num_simulations
    avg_pm = total_pm / num_simulations
    avg_cm_n = total_cm_n / num_simulations
    avg_cm_e = total_cm_e / num_simulations
    avg_cost = total_cost / num_simulations

    return {
        "Total Inspections": total_inspections,
        "Total Preventive Maintenances": total_pm,
        "Total Normal Corrective Maintenances": total_cm_n,
        "Total Emergency Corrective Maintenances": total_cm_e,
        "Total Cost": total_cost,
        "Average Inspections": avg_inspections,
        "Average Preventive Maintenances": avg_pm,
        "Average Normal Corrective Maintenances": avg_cm_n,
        "Average Emergency Corrective Maintenances": avg_cm_e,
        "Average Cost": avg_cost
    }

def run_full_simulation_on_file(file_path, config):
    """
    Executes the full pipeline on a single Excel file and returns results + metadata.
    """
    try:
        # Step 1: Degradation extraction
        degradation_info = take_in(file_path)
        D_0LLs = degradation_info["D_0LLs"]
        degradation_mean = np.log(degradation_info["degradation_mean"])
        degradation_stddev = np.sqrt(np.log(1 + (degradation_info["degradation_stddev"] / degradation_info["degradation_mean"])**2))

        # Step 2: Recovery model
        if config.get("use_accurate_recovery", 1):
            recovery_model = derive_accurate_recovery(file_path, degradation_info["degradation_rates"])
        else:
            recovery_model = derive_recovery(file_path)

        # Step 3: Defect model
        defect_model = derive_defect_probabilities(file_path)

        # Step 4: Simulation
        results = monte(
            time_horizon=config["time_horizon"],
            T_insp=config["T_insp"],
            T_tamp=config["T_tamp"],
            T_step=config["T_step"],
            AL=config["AL"],
            D_0LLs=D_0LLs,
            degradation_mean=degradation_mean,
            degradation_stddev=degradation_stddev,
            e_s_mean=0.0,
            e_s_variance=0.01,
            re_s_m=recovery_model["e_s_mean"],
            re_s_s=recovery_model["e_s_stddev"],
            ILL=config["ILL"],
            IALL=config["IALL"],
            RM=config["RM"],
            RV=config["RV"],
            num_simulations=config["num_simulations"],
            inspection_cost=config["inspection_cost"],
            preventive_maintenance_cost=config["preventive_maintenance_cost"],
            normal_corrective_maintenance_cost=config["normal_corrective_maintenance_cost"],
            emergency_corrective_maintenance_cost=config["emergency_corrective_maintenance_cost"],
            recovery_coefficients=recovery_model,
            defect_coefficients=defect_model
        )

        results.update({
            "file": file_path,
            "used_accurate_recovery": bool(config.get("use_accurate_recovery", 1)),
            "recovery_coefficients": recovery_model,
            "defect_coefficients": defect_model
        })

        return results

    except Exception as e:
        return {
            "file": file_path,
            "error": str(e),
            "status": "failed"
        }
    
simulation_config = {
    # === Core Simulation Control ===
    
    "use_accurate_recovery": 1,         # 1 = Use degradation-adjusted recovery model, 0 = Use simple recovery

    "time_horizon": 3650,              # Total number of days to simulate (e.g., 10 years)
    "T_insp": 30,                      # Inspection interval in days
    "T_tamp": 90,                      # Scheduled tamping interval in days
    "T_step": 1,                       # Time step for simulation loop (should usually be 1)

    # === Maintenance Triggers ===

    "AL": 4.5,                         # DLL_s alert limit (for preventive maintenance)
    "ILL": 0.25,                       # Intermediate Defect Probability Threshold
    "IALL": 0.05,                      # Severe Defect Probability Threshold

    # === Response Dynamics ===

    "RM": 10,                          # Mean delay (in days) to respond to intermediate-level defects
    "RV": 5,                           # Variance of the above response delay

    # === Simulation Volume ===

    "num_simulations": 100,           # Number of independent Monte Carlo simulations per section

    # === Cost Modeling ===

    "inspection_cost": 100,                      # Cost per inspection
    "preventive_maintenance_cost": 200,          # Cost of preventive tamping
    "normal_corrective_maintenance_cost": 500,   # Cost of corrective maintenance for ILL class
    "emergency_corrective_maintenance_cost": 1000 # Cost for emergency response (IALL)
}

def run_simulations_on_batch(file_paths, config):
    """
    Runs the full simulation pipeline on multiple Excel files.

    Parameters:
        file_paths (list of str): Paths to Excel files.
        config (dict): Simulation configuration shared across files.

    Returns:
        list of dicts: Individual simulation results per file.
    """
    results = []
    for path in file_paths:
        print(f"Processing: {path}")
        result = run_full_simulation_on_file(path, config)
        results.append(result)
    return results

def standardize_output(results, config, file_name=None, recovery_coeffs=None, defect_coeffs=None):
    """
    Merges config + results + metadata into a flat structure.

    Parameters:
        results (dict): Output from the monte() function
        config (dict): Input config used for simulation
        file_name (str): Optional name of the Excel file used
        recovery_coeffs (dict): Recovery coefficients used
        defect_coeffs (dict): Defect coefficients used

    Returns:
        dict: Fully flattened and labeled for export.
    """
    output = {
        "file": file_name if file_name else "user_defined",
        "used_accurate_recovery": bool(config.get("use_accurate_recovery", 1)),
    }

    # Merge simulation config
    output.update({f"param_{k}": v for k, v in config.items()})

    # Merge model coefficients
    if recovery_coeffs:
        output.update({f"recovery_{k}": v for k, v in recovery_coeffs.items()})
    if defect_coeffs:
        output.update({f"defect_{k}": v for k, v in defect_coeffs.items()})

    # Merge simulation results
    output.update({f"result_{k}": v for k, v in results.items() if k != "file"})

    return output


def export_results_to_excel(results_list, output_path=None, sheet_names=None):
    import pandas as pd
    from pathlib import Path
    from datetime import datetime    
    """
    Exports a list of standardized simulation result dictionaries to an Excel file.

    Parameters:
        results_list (list of dict): Each dict represents one full simulation result from `standardize_output()`.
        output_path (str, optional): Path to write Excel file. If None, a timestamped name is generated.
        sheet_names (list of str, optional): Sheet names for each result. If None, defaults to "Run_1", "Run_2", ...

    Returns:
        str: Path to the written Excel file (can be used to fetch or download later).
    """
    # Default sheet names if not provided
    if sheet_names is None:
        sheet_names = [f"Run_{i+1}" for i in range(len(results_list))]

    # Auto-generate filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"simulation_results_{timestamp}.xlsx"

    output_path = Path(output_path)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for result, sheet_name in zip(results_list, sheet_names):
            df = pd.DataFrame([result])  # One row per sheet
            df.T.reset_index().to_excel(writer, sheet_name=sheet_name, index=False, header=["Field", "Value"])

    print(f"[✓] Exported simulation results to: {output_path}")
    return str(output_path)

def run_full_simulation_on_file_with_export(file_path, config, output_path=None):
    """
    Full run on one file, including export.
    """
    result = run_full_simulation_on_file(file_path, config)

    if "error" not in result:
        standardized = standardize_output(
            result=result,
            config=config,
            file_name=file_path,
            recovery_coeffs=result["recovery_coefficients"],
            defect_coeffs=result["defect_coefficients"]
        )
        export_results_to_excel([standardized], output_path=output_path)
    else:
        print(f"[✗] Simulation failed: {result['error']}")

    return result

def run_simulations_on_batch_with_export(file_paths, config, output_path=None):
    """
    Batch run across multiple files, with consolidated export.
    """
    results = run_simulations_on_batch(file_paths, config)
    export_ready = []

    for res in results:
        if "error" not in res:
            standardized = standardize_output(
                result=res,
                config=config,
                file_name=res["file"],
                recovery_coeffs=res["recovery_coefficients"],
                defect_coeffs=res["defect_coefficients"]
            )
            export_ready.append(standardized)
        else:
            print(f"[!] Skipping export for failed file: {res['file']} — {res['error']}")

    if export_ready:
        export_results_to_excel(export_ready, output_path=output_path)
    else:
        print("[!] No successful results to export.")

    return results

def run_manual_monte_with_export(config, recovery_coeffs, defect_coeffs, D_0LLs, output_path=None, label=None):
    """
    Executes a standalone Monte Carlo run without data analysis,
    using fully user-specified coefficients and inputs.

    Parameters:
        config (dict): Simulation settings and parameters
        recovery_coeffs (dict): Recovery model coefficients
        defect_coeffs (dict): Defect model coefficients
        D_0LLs (float): Initial post-tamping DLL_s value
        output_path (str, optional): Path to save Excel output
        label (str, optional): Label for identifying run in output

    Returns:
        dict: Simulation results
    """
    try:
        degradation_mean = config.get("degradation_mean")
        degradation_stddev = config.get("degradation_stddev")

        if degradation_mean is None or degradation_stddev is None:
            raise ValueError("config must include 'degradation_mean' and 'degradation_stddev' for standalone mode.")

        results = monte(
            time_horizon=config["time_horizon"],
            T_insp=config["T_insp"],
            T_tamp=config["T_tamp"],
            T_step=config["T_step"],
            AL=config["AL"],
            D_0LLs=D_0LLs,
            degradation_mean=degradation_mean,
            degradation_stddev=degradation_stddev,
            e_s_mean=0.0,
            e_s_variance=0.01,
            re_s_m=recovery_coeffs["e_s_mean"],
            re_s_s=recovery_coeffs["e_s_stddev"],
            ILL=config["ILL"],
            IALL=config["IALL"],
            RM=config["RM"],
            RV=config["RV"],
            num_simulations=config["num_simulations"],
            inspection_cost=config["inspection_cost"],
            preventive_maintenance_cost=config["preventive_maintenance_cost"],
            normal_corrective_maintenance_cost=config["normal_corrective_maintenance_cost"],
            emergency_corrective_maintenance_cost=config["emergency_corrective_maintenance_cost"],
            recovery_coefficients=recovery_coeffs,
            defect_coefficients=defect_coeffs
        )

        # === Bundle Output
        standardized = standardize_output(
            results=results,
            config=config,
            file_name=label if label else "manual_simulation",
            recovery_coeffs=recovery_coeffs,
            defect_coeffs=defect_coeffs
        )

        export_results_to_excel([standardized], output_path=output_path)
        return results

    except Exception as e:
        print(f"[✗] Manual Monte run failed: {e}")
        return {"error": str(e)}

def analyze_file_only(file_path):
    """
    Runs data analysis (degradation, recovery, defect model) on a single Excel file,
    without performing any simulations.

    Parameters:
        file_path (str): Path to Excel file.

    Returns:
        dict: {
            'file': str,
            'D_0LLs': float,
            'degradation_mean': float,
            'degradation_stddev': float,
            'degradation_rates': list,
            'time_intervals': list,
            'recovery_model': dict,
            'defect_model': dict
        }
    """
    try:
        degradation_info = take_in(file_path)
        D_0LLs = degradation_info["D_0LLs"]
        degradation_mean = np.log(degradation_info["degradation_mean"])
        degradation_stddev = np.sqrt(np.log(1 + (degradation_info["degradation_stddev"] / degradation_info["degradation_mean"])**2))

        # Prefer accurate recovery by default
        recovery_model = derive_accurate_recovery(file_path, degradation_info["degradation_rates"])
        defect_model = derive_defect_probabilities(file_path)

        return {
            "file": file_path,
            "D_0LLs": D_0LLs,
            "degradation_mean": degradation_mean,
            "degradation_stddev": degradation_stddev,
            "degradation_rates": degradation_info["degradation_rates"],
            "time_intervals": degradation_info["time_intervals"],
            "recovery_model": recovery_model,
            "defect_model": defect_model
        }

    except Exception as e:
        return {
            "file": file_path,
            "error": str(e),
            "status": "failed"
        }
    
def analyze_files_only(file_paths):
    return [analyze_file_only(path) for path in file_paths]



"""
# SAMPLE RUN, ONE TRACK, 5 YEARS

result = sim_seg(
     time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean,
     degradation_stddev,e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV
 )
print(result)
"""

"""
# MONTE CARLO SIM, ONE TRACK, 5 YEARS
num_simulations = 100  # Number of Monte Carlo simulations
result = monte(
    time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean,
    degradation_stddev,e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV,
    num_simulations=num_simulations,
    inspection_cost=240,
    preventive_maintenance_cost=5000,
    normal_corrective_maintenance_cost=11000,
    emergency_corrective_maintenance_cost=40000
)
print(result)
"""

from pathlib import Path

# Build config dict
manual_config = {
    "time_horizon": 5475,          # 15 years
    "T_insp": 90,                  # Inspection every 3 months
    "T_tamp": 365,                 # PM every year
    "T_step": 1,
    "AL": 2.2,                     # More relaxed preventive limit
    "ILL": 0.4,                    # Moderate CM trigger
    "IALL": 0.15,                  # Higher emergency threshold
    "RM": 35,                      # Response delay (same)
    "RV": 7,
    "num_simulations": 1000,
    "inspection_cost": 240,
    "preventive_maintenance_cost": 3500,
    "normal_corrective_maintenance_cost": 11000,
    "emergency_corrective_maintenance_cost": 40000,
    "degradation_mean": -2.379,
    "degradation_stddev": 0.756
}

# Clean up defect coefficients for compatibility with expected keys
defect_coefficients_clean = {
    "C0": defect_coefficients["C0"],
    "C1": defect_coefficients["C1"],
    "beta": defect_coefficients["b"]
}

# Run it
result = run_manual_monte_with_export(
    config=manual_config,
    recovery_coeffs={
        **recovery_coefficients,
        "e_s_mean": e_LL_mean,
        "e_s_stddev": e_LL_variance
    },
    defect_coeffs=defect_coefficients_clean,
    D_0LLs=D_0LLs,
    label="manual_sim_test"  # Optional, just helps identify the run
)