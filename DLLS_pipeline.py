# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import anderson
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import subprocess
import sys
import xlsxwriter

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

def derive_global_degradation(file_paths):
    import numpy as np
    from scipy.stats import anderson

    all_bs_values = []

    for fp in file_paths:
        try:
            deg_info = take_in(fp)  # Use file-specific logic
            all_bs_values.extend(deg_info["degradation_rates"])
        except Exception as e:
            print(f"[!] Skipping degradation for {fp}: {e}")

    if not all_bs_values:
        raise ValueError("No degradation rates collected from any files.")

    all_bs_values = [b for b in all_bs_values if b > 0]
    if not all_bs_values:
        raise ValueError("No valid (positive) degradation rates found.")

    log_bs = np.log(all_bs_values)
    result = anderson(log_bs, dist="norm")

    if result.statistic < result.critical_values[2]:
        print("Global degradation: lognormal distribution accepted.")
        log_mean = np.mean(log_bs)
        log_std = np.std(log_bs)

        degradation_mean = np.exp(log_mean + 0.5 * log_std**2)
        degradation_stddev = np.sqrt((np.exp(log_std**2) - 1) * np.exp(2 * log_mean + log_std**2))
        use_lognormal = True
    else:
        print("Global degradation: normal distribution assumed.")
        degradation_mean = np.mean(all_bs_values)
        degradation_stddev = np.std(all_bs_values)
        use_lognormal = False

    return {
        "degradation_mean": degradation_mean,
        "degradation_stddev": degradation_stddev,
        "use_lognormal": use_lognormal,
        "all_bs": all_bs_values  # keep for recovery if needed
    }

def derive_global_recovery(file_paths):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    all_X = []
    all_Y = []

    for fp in file_paths:
        try:
            df = pd.read_excel(fp)
            if df.empty:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            if "DLL_s" not in df.columns:
                if "A" in df.columns:
                    df = df.rename(columns={"A": "DLL_s"})
                elif "G" in df.columns:
                    df = df.rename(columns={"G": "DLL_s"})
                else:
                    continue

            required_cols = ["DLL_s", "Tamping Performed", "Tamping Type"]
            if not all(col in df.columns for col in required_cols):
                continue

            if df["DLL_s"].std() < 1e-6 or (df["DLL_s"] < 0).any():
                continue

            tamping_events = df[df["Tamping Performed"] == 1]
            if len(tamping_events) < 4:
                continue

            DLL_s = df["DLL_s"].values
            recovery_vals = []
            recovery_indices = []

            for idx in tamping_events.index:
                if idx + 1 < len(df):
                    delta = DLL_s[idx] - DLL_s[idx + 1]
                    recovery_vals.append(delta)
                    recovery_indices.append(idx)

            if len(recovery_vals) < 3:
                continue

            tap_df = df.loc[recovery_indices].copy()
            X = tap_df[["DLL_s", "Tamping Type"]].copy()
            X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})
            X["Interaction"] = X["DLL_s"] * X["Tamping Type"]

            all_X.append(X)
            all_Y.append(np.array(recovery_vals))

        except Exception as e:
            print(f"[!] Skipping recovery from {fp}: {e}")

    if not all_X or not all_Y:
        raise ValueError("No valid recovery data extracted from input files.")

    X_full = pd.concat(all_X, ignore_index=True).values
    Y_full = np.concatenate(all_Y)

    model = LinearRegression()
    model.fit(X_full, Y_full)

    residuals = Y_full - model.predict(X_full)
    ad_result = anderson(residuals, dist="norm")

    normal_resid = ad_result.statistic < ad_result.critical_values[2]
    if normal_resid:
        print("Global recovery residuals follow normal distribution.")
    else:
        print("Global recovery residuals DO NOT follow normal distribution.")

    return {
        "alpha_1": model.intercept_,
        "beta_1": model.coef_[0],
        "beta_2": model.coef_[1],
        "beta_3": model.coef_[2],
        "e_s_mean": np.mean(residuals),
        "e_s_stddev": np.std(residuals)
    }

    
def derive_global_accurate_recovery_per_file_bs(file_paths):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    all_X = []
    all_Y = []

    for fp in file_paths:
        try:
            # Get degradation info for this file
            deg_info = take_in(fp)
            file_bs_list = deg_info["degradation_rates"]
            file_avg_bs = np.mean([b for b in file_bs_list if b > 0])

            df = pd.read_excel(fp)
            if df.empty:
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            if "DLL_s" not in df.columns:
                if "A" in df.columns:
                    df = df.rename(columns={"A": "DLL_s"})
                elif "G" in df.columns:
                    df = df.rename(columns={"G": "DLL_s"})
                else:
                    continue

            required_cols = ["DLL_s", "Tamping Performed", "Tamping Type", "Date"]
            if not all(col in df.columns for col in required_cols):
                continue

            if df["DLL_s"].std() < 1e-6 or (df["DLL_s"] < 0).any():
                continue

            tamping_events = df[df["Tamping Performed"] == 1]
            tamp_indices = tamping_events.index.tolist()
            if len(tamp_indices) < 4:
                continue

            DLL_s = df["DLL_s"].values

            adjusted_R_vals = []
            valid_indices = []

            for i in range(len(tamp_indices) - 1):
                idx = tamp_indices[i]
                next_idx = tamp_indices[i + 1]

                if next_idx >= len(df):
                    continue

                t1 = df.loc[idx, "Date"]
                t2 = df.loc[next_idx, "Date"]
                interval = (t2 - t1).days

                if interval <= 0:
                    continue

                observed_R = DLL_s[idx] - DLL_s[next_idx]
                adjusted_R = observed_R + (file_avg_bs * interval)

                adjusted_R_vals.append(adjusted_R)
                valid_indices.append(idx)

            if len(adjusted_R_vals) < 3:
                continue

            tap_df = df.loc[valid_indices].copy()
            X = tap_df[["DLL_s", "Tamping Type"]].copy()
            X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})
            X["Interaction"] = X["DLL_s"] * X["Tamping Type"]

            all_X.append(X)
            all_Y.append(np.array(adjusted_R_vals))

        except Exception as e:
            print(f"[!] Skipping accurate recovery from {fp}: {e}")

    if not all_X or not all_Y:
        raise ValueError("No valid recovery data found for accurate model.")

    X_full = pd.concat(all_X, ignore_index=True).values
    Y_full = np.concatenate(all_Y)

    model = LinearRegression()
    model.fit(X_full, Y_full)

    residuals = Y_full - model.predict(X_full)
    ad_result = anderson(residuals, dist="norm")

    normal_resid = ad_result.statistic < ad_result.critical_values[2]
    if normal_resid:
        print("Accurate recovery residuals follow normal distribution.")
    else:
        print("Accurate recovery residuals DO NOT follow normal distribution.")

    return {
        "alpha_1": model.intercept_,
        "beta_1": model.coef_[0],
        "beta_2": model.coef_[1],
        "beta_3": model.coef_[2],
        "e_s_mean": np.mean(residuals),
        "e_s_stddev": np.std(residuals)
    }

def derive_global_defect_probabilities(file_paths):
    import pandas as pd
    import numpy as np
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    all_dll = []
    all_defects = []

    for fp in file_paths:
        try:
            df = pd.read_excel(fp)
            if df.empty:
                continue

            # Rename DLL_s if needed
            if "DLL_s" not in df.columns:
                if "A" in df.columns:
                    df = df.rename(columns={"A": "DLL_s"})
                elif "G" in df.columns:
                    df = df.rename(columns={"G": "DLL_s"})
                else:
                    continue

            required = ["DLL_s", "Defect"]
            if not all(c in df.columns for c in required):
                continue

            if df["DLL_s"].std() < 1e-6 or (df["DLL_s"] < 0).any():
                continue

            if not set(df["Defect"].unique()).issubset({0, 1, 2}):
                continue

            all_dll.extend(df["DLL_s"].tolist())
            all_defects.extend(df["Defect"].tolist())

        except Exception as e:
            print(f"[!] Skipping defect data from {fp}: {e}")

    if len(all_dll) < 10:
        raise ValueError("Not enough total defect samples for global model.")

    X = pd.DataFrame({"DLL_s": all_dll})
    Y = pd.Series(all_defects)

    try:
        model = OrderedModel(Y, X, distr="logit")
        results = model.fit(method="lbfgs", disp=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fit global defect model: {e}")

    if not results.mle_retvals.get("converged", False):
        raise RuntimeError("Global defect model did not converge.")

    print("Global defect model trained.")
    print(results.summary())

    return results  # Return full model object
    

def take_in(file_path):
# deg calc 
    import pandas as pd
    import numpy as np
    from scipy.stats import anderson

    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("Input Excel file is empty.")

    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Could not find DLL_s column.")

    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    tamping_indices = df[df["Tamping Performed"] == 1].index.tolist()
    if len(tamping_indices) < 2:
        raise ValueError("Need at least two tamping events to calculate degradation rates.")

    degradation_rates = []
    time_intervals = []

    for i in range(len(tamping_indices) - 1):
        start_idx = tamping_indices[i] + 1
        end_idx = tamping_indices[i + 1]

        if end_idx <= start_idx or end_idx >= len(df):
            continue

        sub_df = df.loc[start_idx:end_idx]
        if sub_df.empty:
            continue

        dll_start = sub_df["DLL_s"].iloc[0]
        dll_end = sub_df["DLL_s"].iloc[-1]
        days = (sub_df["Date"].iloc[-1] - sub_df["Date"].iloc[0]).days

        if days <= 0:
            continue

        b_s = (dll_end - dll_start) / days
        degradation_rates.append(b_s)
        time_intervals.append(days)

    positive_b_s = [x for x in degradation_rates if x > 0]
    if len(positive_b_s) == 0:
        raise ValueError("No positive degradation rates found.")

    log_bs = np.log(positive_b_s)
    result = anderson(log_bs, dist="norm")

    if result.statistic < result.critical_values[2]:
        # print("b_s values follow a lognormal distribution.")
        log_mean = np.mean(log_bs)
        log_std = np.std(log_bs)

        degradation_mean = np.exp(log_mean + 0.5 * log_std**2)
        degradation_stddev = np.sqrt((np.exp(log_std**2) - 1) * np.exp(2 * log_mean + log_std**2))
    else:
        # print("b_s values do not follow a lognormal distribution. Assuming normal.")
        degradation_mean = np.mean(degradation_rates)
        degradation_stddev = np.std(degradation_rates)

    most_recent_tamping = df[df["Tamping Performed"] == 1].iloc[-1]
    most_recent_index = most_recent_tamping.name

    if most_recent_index + 1 < len(df):
        D_0LLs = df.iloc[most_recent_index + 1]["DLL_s"]
    else:
        D_0LLs = most_recent_tamping["DLL_s"]

    # print(f"Degradation Mean: {degradation_mean}")
    # print(f"Degradation Stddev: {degradation_stddev}")
    # print(f"D_0LLs (Initial DLL_s after last tamping): {D_0LLs}")

    return {
        "D_0LLs": D_0LLs,
        "degradation_mean": degradation_mean,
        "degradation_stddev": degradation_stddev,
        "degradation_rates": degradation_rates,
        "time_intervals": time_intervals,
        "use_lognormal": result.statistic < result.critical_values[2]  
    }


def derive_recovery(file_path):
    # recovery 
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string path to an Excel file.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    if df.empty:
        raise ValueError("The input Excel file is empty.")

    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Missing DLL_s column. Expected one of: DLL_s, A, G.")

    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    if df["DLL_s"].std() < 1e-6:
        raise ValueError("DLL_s column has near-zero variance. Regression will be unstable.")

    if (df["DLL_s"] < 0).any():
        raise ValueError("DLL_s contains negative values, which is not physically valid.")

    tamping_events = df[df["Tamping Performed"] == 1]
    if len(tamping_events) < 4:
        raise ValueError("Not enough tamping events (min 4 required) for regression.")

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

    tamping_events = df.loc[recovery_indices]
    X = tamping_events[["DLL_s", "Tamping Type"]].copy()
    X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})
    X["Interaction"] = X["DLL_s"] * X["Tamping Type"]

    Y = np.array(R_LL_s)
    X = X.values
    
    model = LinearRegression()
    model.fit(X, Y)

    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    # print(f"Derived Recovery Coefficients:")
    # print(f"α₁ (Intercept): {alpha_1}")
    # print(f"β₁ (DLL_s): {beta_1}")
    # print(f"β₂ (Tamping Type): {beta_2}")
    # print(f"β₃ (Interaction): {beta_3}")

    residuals = Y - model.predict(X)
    ad_test_result = anderson(residuals, dist="norm")

    if ad_test_result.statistic < ad_test_result.critical_values[2]:
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        # print("Residuals follow normal distribution.")
    else:
        # print("Residuals DO NOT follow a normal distribution. Simulation will still proceed.")
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
    # subtracts degradation between events
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

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

    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        raise ValueError("Failed to convert 'Date' to datetime format.")

    if (df["DLL_s"] < 0).any():
        raise ValueError("DLL_s contains negative values.")

    if df["DLL_s"].std() < 1e-6:
        raise ValueError("DLL_s has too little variance to train a meaningful model.")

    tamping_events = df[df["Tamping Performed"] == 1]
    if len(tamping_events) < 4:
        raise ValueError("Not enough tamping events (min 4 required).")

    average_bs = np.mean(bs_values)

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

    # print("Derived Accurate Recovery Coefficients:")
    # print(f"α₁ (Intercept): {alpha_1}")
    # print(f"β₁ (DLL_s): {beta_1}")
    # print(f"β₂ (Tamping Type): {beta_2}")
    # print(f"β₃ (Interaction): {beta_3}")

    residuals = Y - model.predict(X)
    ad_test_result = anderson(residuals, dist="norm")

    if ad_test_result.statistic < ad_test_result.critical_values[2]:
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        # print("Residuals follow normal distribution.")
    else:
        # print("Residuals do NOT follow normal distribution. Using estimated mean/std anyway.")
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
    import pandas as pd
    import numpy as np
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    if not isinstance(file_path, str):
        raise ValueError("file_path must be a valid Excel file path.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df.empty:
        raise ValueError("Input data file is empty.")

    if "DLL_s" not in df.columns:
        if "A" in df.columns:
            df = df.rename(columns={"A": "DLL_s"})
        elif "G" in df.columns:
            df = df.rename(columns={"G": "DLL_s"})
        else:
            raise ValueError("Missing DLL_s column. Expected one of: DLL_s, A, G.")

    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    if df["DLL_s"].std() < 1e-6:
        raise ValueError("DLL_s variance too low. Model will be unstable.")

    if (df["DLL_s"] < 0).any():
        raise ValueError("DLL_s contains negative values, which is invalid.")

    if "Defect" not in df.columns:
        raise ValueError("Missing 'Defect' column.")

    if not set(df["Defect"].unique()).issubset({0, 1, 2}):
        raise ValueError("Defect values must be only 0, 1, or 2.")

    class_counts = df["Defect"].value_counts()
    if class_counts.min() < 2:
        raise ValueError("One or more defect classes are underrepresented (< 2 samples).")

    X = df[["DLL_s"]]
    Y = df["Defect"]

    try:
        model = OrderedModel(Y, X, distr="logit")
        results = model.fit(method="lbfgs", disp=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fit ordinal model: {e}")

    if not results.mle_retvals.get("converged", False):
        raise RuntimeError("Ordinal model did not converge.")

    print("Derived Defect Probability Parameters (via model object):")
    print(results.summary())

    return results  

# Example 
# coefficients = derive_defect_probabilities("mock_data_file.xlsx")
# print(coefficients)

def degrade(current_DLL_s, b_s, time, e_s):
    
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

def defect(current_DLL_s, defect_model):
    import pandas as pd
    import numpy as np

    if current_DLL_s < 0:
        raise ValueError("current_DLL_s must be ≥ 0.")

    if defect_model is None or not hasattr(defect_model, 'predict'):
        raise ValueError("Invalid defect model provided. Expected a trained OrderedModelResults object.")

    try:
        X = pd.DataFrame({"DLL_s": [current_DLL_s]})
        preds = defect_model.predict(X)

        # The result is a DataFrame with columns [0, 1, 2] for classes
        pred_row = preds.iloc[0]

        return {
            "P0": pred_row.get(0, 0.0),
            "P1": pred_row.get(1, 0.0),
            "P2": pred_row.get(2, 0.0)
        }

    except Exception as e:
        raise RuntimeError(f"Failed to compute defect probabilities: {e}")



def sim_seg(time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, lognormal_mean, lognormal_stddev,
            e_s_mean, e_s_s, re_s_m, re_s_s, ILL, IALL, RM, RV,
            recovery_coefficients, defect_model, use_lognormal_bs: bool):

    if T_insp % T_step != 0 or T_tamp % T_step != 0:
        raise ValueError("T_insp and T_tamp must be multiples of T_step for aligned intervals.")

    t = 1
    tn = 0
    Npm = Ncm_n = Ncm_e = Ninsp = total_response_delay = 0

    # Sample initial degradation rate
    if use_lognormal_bs:
        b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)
    else:
        b_s = np.random.normal(loc=lognormal_mean, scale=lognormal_stddev)

    while t <= time_horizon:
        t += T_step
        e_s = np.random.normal(e_s_mean, e_s_s)
        DLL_s_t = max(D_0LLs + b_s * (t - tn) + e_s, 0)

        if t > time_horizon:
            break

        # Preventive maintenance decision
        if t % T_tamp == 0 and DLL_s_t > AL:
            D_0LLs = recovery(DLL_s_t, recovery_coefficients, tamping_type=1, re_s_m=re_s_m, re_s_s=re_s_s)
            Npm += 1
            tn = t
            b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev) if use_lognormal_bs else \
                np.random.normal(loc=lognormal_mean, scale=lognormal_stddev)
            continue

        # Inspection event
        if t % T_insp == 0:
            Ninsp += 1
            probs = defect(DLL_s_t, defect_model)  # <- pass full model now
            P2 = probs["P2"]
            P3 = probs["P3"]

            # Emergency corrective
            if P3 > IALL:
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, tamping_type=0, re_s_m=re_s_m, re_s_s=re_s_s)
                Ncm_e += 1
                tn = t
                continue

            # Normal corrective
            if P2 > ILL:
                response_time = int(np.random.normal(loc=RM, scale=RV))
                response_time = max(0, response_time)
                total_response_delay += response_time
                t += response_time

                delayed_e_s = np.random.normal(e_s_mean, e_s_s)
                DLL_s_t = degrade(DLL_s_t, b_s, response_time, delayed_e_s)

                D_0LLs = recovery(DLL_s_t, recovery_coefficients, tamping_type=0, re_s_m=re_s_m, re_s_s=re_s_s)
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
    recovery_coefficients, defect_model, use_lognormal_bs: bool
):

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
            recovery_coefficients, defect_model,  # ← pass model directly
            use_lognormal_bs
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
        "Average Cost": avg_cost,
        "Final D0LL_s": D_0LLs
    }

def run_full_simulation_on_file(file_path, config):
    try:
        degradation_info = take_in(file_path)
        D_0LLs = degradation_info["D_0LLs"]

        if degradation_info["use_lognormal"]:
            degradation_mean = np.log(degradation_info["degradation_mean"])
            degradation_stddev = np.sqrt(np.log(1 + (degradation_info["degradation_stddev"] / degradation_info["degradation_mean"])**2))
        else:
            degradation_mean = degradation_info["degradation_mean"]
            degradation_stddev = degradation_info["degradation_stddev"]

        if config.get("use_accurate_recovery", 1):
            recovery_model = derive_accurate_recovery(file_path, degradation_info["degradation_rates"])
        else:
            recovery_model = derive_recovery(file_path)

        defect_model = derive_defect_probabilities(file_path)

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
            defect_model=defect_model,  # << updated key
            use_lognormal_bs=degradation_info["use_lognormal"]
        )

        results.update({
            "file": file_path,
            "used_accurate_recovery": bool(config.get("use_accurate_recovery", 1)),
            "used_lognormal_bs": degradation_info["use_lognormal"],
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
    
def run_simulations_on_batch(file_paths, config):
    from pathlib import Path

    dll_starts = {}

    print("Extracting initial D_0LLs per file...")
    for fp in file_paths:
        try:
            deg_info = take_in(fp)
            dll_starts[fp] = deg_info["D_0LLs"]
        except Exception as e:
            print(f"[!] Skipping {fp}: {e}")

    if not dll_starts:
        raise ValueError("No valid D_0LLs extracted from any files.")

    print("Training global degradation model...")
    deg_model = derive_global_degradation(file_paths)
    use_lognormal = deg_model["use_lognormal"]

    if use_lognormal:
        degradation_mean = np.log(deg_model["degradation_mean"])
        degradation_stddev = np.sqrt(np.log(1 + (deg_model["degradation_stddev"] / deg_model["degradation_mean"])**2))
    else:
        degradation_mean = deg_model["degradation_mean"]
        degradation_stddev = deg_model["degradation_stddev"]

    print("Training global recovery model...")
    if config.get("use_accurate_recovery", 1):
        recovery_model = derive_global_accurate_recovery_per_file_bs(file_paths)
    else:
        recovery_model = derive_global_recovery(file_paths)

    print("Training global defect model...")
    defect_model = derive_global_defect_probabilities(file_paths)

    results = []
    for path in file_paths:
        try:
            print(f"Simulating: {path}")
            result = monte(
                time_horizon=config["time_horizon"],
                T_insp=config["T_insp"],
                T_tamp=config["T_tamp"],
                T_step=config["T_step"],
                AL=config["AL"],
                D_0LLs=dll_starts[path],
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
                defect_model=defect_model,  # << FIXED KEY
                use_lognormal_bs=use_lognormal
            )

            result.update({
                "file": path,
                "used_accurate_recovery": bool(config.get("use_accurate_recovery", 1)),
                "used_lognormal_bs": use_lognormal,
                "recovery_coefficients": recovery_model,
                "defect_coefficients": defect_model  # ok to keep for metadata
            })

            results.append(result)

        except Exception as e:
            results.append({
                "file": path,
                "error": str(e),
                "status": "failed"
            })

    return results

def standardize_output(results, config, file_name=None, recovery_coeffs=None, defect_model=None):
    output = {
        "file": file_name if file_name else "user_defined",
        "used_lognormal_bs": results.get("used_lognormal_bs", True),
        "used_accurate_recovery": bool(config.get("use_accurate_recovery", 1)),
    }
    
    output.update({f"param_{k}": v for k, v in config.items()})

    if recovery_coeffs:
        output.update({f"recovery_{k}": v for k, v in recovery_coeffs.items()})

    if defect_model:
        try:
            model_params = defect_model.params.to_dict()
            output.update({f"defect_{k}": v for k, v in model_params.items()})
        except Exception:
            output["defect_model_repr"] = str(defect_model)[:500] 

    output.update({f"result_{k}": v for k, v in results.items() if k != "file"})

    return output


def export_results_to_excel(results_list, output_path=None, sheet_names=None):
    import pandas as pd
    from pathlib import Path
    from datetime import datetime    

    if sheet_names is None:
        sheet_names = [f"Run_{i+1}" for i in range(len(results_list))]

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"simulation_results_{timestamp}.xlsx"

    output_path = Path(output_path)

    all_rows = pd.DataFrame(results_list)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for result, sheet_name in zip(results_list, sheet_names):
            df = pd.DataFrame([result])
            df.T.reset_index().to_excel(writer, sheet_name=sheet_name, index=False, header=["Field", "Value"])

        numeric_cols = all_rows.select_dtypes(include="number").columns
        summary_df = all_rows[numeric_cols].mean().reset_index()
        summary_df.columns = ["Metric", "Average"]
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Exported simulation results + summary to: {output_path}")
    return str(output_path)

def run_full_simulation_on_file_with_export(file_path, config, output_path=None):
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
        print(f"Simulation failed: {result['error']}")

    return result

def run_simulations_on_batch_with_export(file_paths, config, output_path=None):
# global 
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
            print(f"Skipping export for failed file: {res['file']} — {res['error']}")

    if export_ready:
        export_results_to_excel(export_ready, output_path=output_path)
    else:
        print("No successful results to export.")

    return results


def run_manual_monte_with_export(use_lognormal_bs, config, recovery_coeffs, defect_model, D_0LLs, output_path=None, label=None):
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
            defect_model=defect_model,  # << updated key
            use_lognormal_bs=use_lognormal_bs
        )

        standardized = standardize_output(
            results=results,
            config=config,
            file_name=label if label else "manual_simulation",
            recovery_coeffs=recovery_coeffs,
            defect_coeffs=defect_model  # this is just metadata
        )

        export_results_to_excel([standardized], output_path=output_path)
        return results

    except Exception as e:
        print(f"Manual Monte run failed: {e}")
        return {"error": str(e)}

def analyze_file_only(file_path):
    try:
        degradation_info = take_in(file_path)
        D_0LLs = degradation_info["D_0LLs"]
        degradation_mean = np.log(degradation_info["degradation_mean"])
        degradation_stddev = np.sqrt(np.log(1 + (degradation_info["degradation_stddev"] / degradation_info["degradation_mean"])**2))

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
    try:
        print("[*] Running global model analysis only (no simulation)...")

        # Global degradation
        global_deg = derive_global_degradation(file_paths)
        degradation_mean = (
            np.log(global_deg["degradation_mean"]) if global_deg["use_lognormal"]
            else global_deg["degradation_mean"]
        )
        degradation_stddev = (
            np.sqrt(np.log(1 + (global_deg["degradation_stddev"] / global_deg["degradation_mean"])**2))
            if global_deg["use_lognormal"]
            else global_deg["degradation_stddev"]
        )

        # Global recovery (accurate or not)
        global_recovery = derive_global_accurate_recovery_per_file_bs(file_paths)

        # Global defect model
        global_defect = derive_global_defect_probabilities(file_paths)

        return {
            "degradation_mean": degradation_mean,
            "degradation_stddev": degradation_stddev,
            "use_lognormal_bs": global_deg["use_lognormal"],
            "recovery_model": global_recovery,
            "defect_model": global_defect
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

from pathlib import Path

manual_config = {
    "time_horizon": 5475,         
    "T_insp": 90,                  
    "T_tamp": 60,                 
    "T_step": 1,
    "AL": 1.3,                    
    "ILL": 0.75,                   
    "IALL": 0.05,                  
    "RM": 35,                      
    "RV": 7,
    "num_simulations": 10000,
    "inspection_cost": 240,
    "preventive_maintenance_cost": 3500,
    "normal_corrective_maintenance_cost": 11000,
    "emergency_corrective_maintenance_cost": 40000,
    "degradation_mean": -5.11,
    "degradation_stddev": 0.75
}



config = {
    "use_accurate_recovery": 1,         # Use degradation-aware recovery model
    "time_horizon": 5475,               # 15 years
    "T_insp": 90,                       # Inspection every 3 months
    "T_tamp": 60,                       # Preventive maintenance every 2 months
    "T_step": 1,
    "AL": 1.3,                          # Alert limit for PM
    "ILL": 0.4,                        # Prob. threshold for normal CM
    "IALL": 0.05,                       # Prob. threshold for emergency CM
    "RM": 35,                           # Response mean delay
    "RV": 7,                            # Response std delay
    "num_simulations": 10000,          # Number of Monte Carlo runs
    "inspection_cost": 240,
    "preventive_maintenance_cost": 3500,
    "normal_corrective_maintenance_cost": 11000,
    "emergency_corrective_maintenance_cost": 40000
}


def collect_excel_files(directory_path):
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"[✗] Directory not found or invalid: {directory_path}")

    excel_files = list(dir_path.glob("*.xlsx"))
    if not excel_files:
        raise ValueError(f"[!] No Excel (.xlsx) files found in: {directory_path}")

    return [str(file) for file in excel_files]

"""
result = monte(
    time_horizon=15 * 365,
    T_insp=120,
    T_tamp=365,
    T_step=1,
    AL=1.5,
    D_0LLs=0.352,
    degradation_mean=-2.379,
    degradation_stddev=0.756,
    e_s_mean=0.0,
    e_s_variance=np.sqrt(0.041),
    re_s_m=0.0,
    re_s_s=np.sqrt(0.15),
    ILL=0.75,
    IALL=0.05,
    RM=35,
    RV=7,
    num_simulations=10000,
    inspection_cost=240,
    preventive_maintenance_cost=5000,
    normal_corrective_maintenance_cost=11000,
    emergency_corrective_maintenance_cost=40000,
    recovery_coefficients={
        "alpha_1": -0.269,
        "beta_1": 0.51,
        "beta_2": 0.207,
        "beta_3": -0.043
    },
    defect_coefficients={
        "C0": 9.1875,
        "C1": 13.39,
        "beta": -4.7712
    },
    use_lognormal_bs=True
)

print(result)

"""

directory_path = r"C:\Users\13046\synthetic_track_dataset"

file_paths = collect_excel_files(directory_path)

results = run_simulations_on_batch_with_export(file_paths, config=config, output_path="results_summary.xlsx")