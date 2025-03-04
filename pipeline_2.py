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
    "C_0": 9.1875,  # Coefficient for IL level
    "C_1": 13.39,   # Coefficient for IAL level
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

"""
PARAMETERS END
"""

def take_in(file_path):
    """
    Reads an Excel file and calculates constants and parameters related to degradation and recovery.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Calculated constants and parameters.
    """
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Check if either "DLL_s" or "A" exists
    if "DLL_s" not in df.columns and "A" not in df.columns:
        raise ValueError(f"Input file must contain either 'DLL_s' or 'A' as a column.")

    # If "A" is present but "DLL_s" is not, rename "A" to "DLL_s"
    if "DLL_s" not in df.columns and "A" in df.columns:
        df = df.rename(columns={"A": "DLL_s"})  # Rename internally

    # Ensure other necessary columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Sort by date to ensure chronological order
    df = df.sort_values(by="Date")
    df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is datetime

    # Calculate time intervals (in days)
    df["Time Interval"] = (df["Date"] - df["Date"].shift(1)).dt.days

    # Identify ranges for degradation rate calculations
    tamping_indices = df[df["Tamping Performed"] == 1].index
    degradation_periods = []

    for i in range(len(tamping_indices) - 1):
        start = tamping_indices[i] + 1  # Start the day after tamping
        end = tamping_indices[i + 1]  # End on the next tamping day
        degradation_periods.append(df.loc[start:end])

    # Calculate degradation rates for each period
    degradation_rates = []
    time_intervals = []

    for period in degradation_periods:
        period["Degradation Rate"] = period["DLL_s"].diff() / period["Time Interval"]
        degradation_rates.extend(period["Degradation Rate"].dropna().tolist())
        time_intervals.extend(period["Time Interval"].dropna().tolist())

    # Test for lognormal distribution
    log_dll_values = np.log(df["DLL_s"][df["DLL_s"] > 0])  
    result = anderson(log_dll_values, dist="norm")

    if result.significance_level[0] > 0.05:  
        print("DLL_s values follow a lognormal distribution.")
        lognormal_mean = np.mean(log_dll_values) 
        lognormal_stddev = np.std(log_dll_values)  

        degradation_mean = np.exp(lognormal_mean + 0.5 * lognormal_stddev**2)
        degradation_stddev = np.sqrt((np.exp(lognormal_stddev**2) - 1) * np.exp(2 * lognormal_mean + lognormal_stddev**2))
    else:
        print("DLL_s values do not follow a lognormal distribution. Assuming normal distribution.")
        degradation_mean = np.mean(df["DLL_s"])
        degradation_stddev = np.std(df["DLL_s"])

    print(f"Degradation Mean: {degradation_mean}")
    print(f"Degradation Stddev: {degradation_stddev}")

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
    # NOTE, the inherent degradation, by nature of observation periods, are included in RLL_s calculation
    # paper does not mention???
    """
    Derives recovery coefficients from the provided dataset using regression and validates res normality.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Derived recovery coefficients {alpha_1, beta_1, beta_2, beta_3} and error term stats {mean, stddev}.
    """
    df = pd.read_excel(file_path)

    # Check if either "DLL_s" or "A" exists
    if "DLL_s" not in df.columns and "A" not in df.columns:
        raise ValueError(f"Input file must contain either 'DLL_s' or 'A' as a column.")

    # If "A" is present but "DLL_s" is not, rename "A" to "DLL_s"
    if "DLL_s" not in df.columns and "A" in df.columns:
        df = df.rename(columns={"A": "DLL_s"})  # Rename internally

    # Ensure other necessary columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Tamping rows
    tamping_events = df[df["Tamping Performed"] == 1]

    # Ensure there are enough points
    if len(tamping_events) < 4:
        raise ValueError("Not enough tamping events to perform regression.")

    # Calculate R_LL_s (difference before and after recovery)
    R_LL_s = []
    DLL_s_values = df["DLL_s"].values

    for index in tamping_events.index:
        if index + 1 < len(df):
            R_LL_s.append(DLL_s_values[index] - DLL_s_values[index + 1])

    tamping_events = tamping_events.iloc[:-1]  # Exclude last tamping event if no pair exists

    # Prepare regression input
    X = tamping_events[["DLL_s", "Tamping Type"]].copy()
    X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})  # Map tamping type: 1 -> 0 (partial), 2 -> 1 (complete)
    X["Interaction"] = X["DLL_s"] * X["Tamping Type"]  # Interaction term

    X = X.values
    Y = np.array(R_LL_s)

    # regression
    model = LinearRegression()
    model.fit(X, Y)

    # Extract coefficients
    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    print(f"Derived Recovery Coefficients:")
    print(f"a1 (Intercept): {alpha_1}")
    print(f"b1 (Coefficient for DLL_s): {beta_1}")
    print(f"b2 (Coefficient for Tamping Type): {beta_2}")
    print(f"b3 (Interaction Coefficient): {beta_3}")

    # Calculate residuals
    residuals = Y - model.predict(X)

    # Test residuals 
    ad_test_result = anderson(residuals, dist='norm')

    if ad_test_result.significance_level[2] > 0.05:  # Check p-value 
        print("Residuals follow a normal distribution.")
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        print(f"Error Term Mean (e_s_mean): {residual_mean}")
        print(f"Error Term Stddev (e_s_stddev): {residual_stddev}")
    else:
        print("Residuals do NOT follow a normal distribution. Error term calculation is invalid.")
        residual_mean = None
        residual_stddev = None

    return {
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "e_s_mean": residual_mean,
        "e_s_stddev": residual_stddev
    }


# This is the true method 
def derive_accurate_recovery(file_path, bs_values):
    """
    Derives recovery coefficients from the provided dataset using regression,
    adjusting the observed recovery effect by adding back the expected degradation
    over the recovery period. This expected degradation is computed using the average
    of the provided daily degradation rates (bs_values). That is, if a tamping event
    shows an observed recovery of R_LL_s over an interval T, then the adjusted recovery
    is R_LL_s + (average_b_s * T).

    Parameters:
        file_path (str): Path to the Excel file containing track data.
        bs_values (array-like): Array (or similar) of daily degradation rates (b_s) as
                                computed in take_in.

    Returns:
        dict: Derived recovery coefficients {alpha_1, beta_1, beta_2, beta_3} and error term stats {mean, stddev}.
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    # Load the Excel file
    df = pd.read_excel(file_path)

    # Check if either "DLL_s" or "A" exists
    if "DLL_s" not in df.columns and "A" not in df.columns:
        raise ValueError(f"Input file must contain either 'DLL_s' or 'A' as a column.")

    # If "A" is present but "DLL_s" is not, rename "A" to "DLL_s"
    if "DLL_s" not in df.columns and "A" in df.columns:
        df = df.rename(columns={"A": "DLL_s"})  # Rename internally

    # Ensure other necessary columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Convert Date column to datetime if not already done
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter for tamping events
    tamping_events = df[df["Tamping Performed"] == 1]
    if len(tamping_events) < 4:
        raise ValueError("Not enough tamping events to perform regression.")

    # Compute the average daily degradation rate from bs_values
    average_bs = np.mean(bs_values)
    
    # Compute the adjusted recovery effect (R_LL_s_adjusted) for each tamping pair.
    # For each tamping event (except the last one), calculate:
    # observed_R = DLL_s (at tamping) - DLL_s (at next measurement)
    # time_interval = days between the two dates (using the same method as in take_in)
    # adjusted_R = observed_R + (average_bs * time_interval)
    DLL_s_values = df["DLL_s"].values
    R_LL_s_adjusted = []
    tamping_indices = tamping_events.index.tolist()  # indices of tamping events

    for i in range(len(tamping_indices) - 1):
        idx = tamping_indices[i]
        next_idx = tamping_indices[i + 1]
        
        # Observed recovery effect from tamping (improvement in DLL_s)
        observed_R = DLL_s_values[idx] - DLL_s_values[next_idx]
        
        # Determine the time interval (in days) between these two measurements
        t1 = df.loc[idx, "Date"]
        t2 = df.loc[next_idx, "Date"]
        interval = (t2 - t1).days
        
        # Adjust the recovery effect by adding back the expected degradation
        # over this interval (average_bs * interval)
        adjusted_R = observed_R + (average_bs * interval)
        R_LL_s_adjusted.append(adjusted_R)

    # We use tamping events (except the last one) as the observations for regression.
    tamping_events = tamping_events.iloc[:-1]

    # Prepare regression input.
    # Use "DLL_s" and "Tamping Type" as predictors, with an interaction term.
    X = tamping_events[["DLL_s", "Tamping Type"]].copy()
    # Map Tamping Type: 1 -> 0 (partial), 2 -> 1 (complete)
    X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})
    X["Interaction"] = X["DLL_s"] * X["Tamping Type"]
    X = X.values

    # The adjusted recovery effects become the response variable Y.
    Y = np.array(R_LL_s_adjusted)

    # Perform linear regression on the adjusted recovery effects.
    model = LinearRegression()
    model.fit(X, Y)

    # Extract regression coefficients
    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    print("Derived Accurate Recovery Coefficients:")
    print(f"alpha_1 (Intercept): {alpha_1}")
    print(f"beta_1 (Coefficient for DLL_s): {beta_1}")
    print(f"beta_2 (Coefficient for Tamping Type): {beta_2}")
    print(f"beta_3 (Interaction Coefficient): {beta_3}")

    # Compute residuals and test for normality of errors
    residuals = Y - model.predict(X)
    ad_test_result = anderson(residuals, dist='norm')
    if ad_test_result.significance_level[2] > 0.05:
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        print("Residuals follow a normal distribution.")
        print(f"Residual Mean: {residual_mean}")
        print(f"Residual Stddev: {residual_stddev}")
    else:
        print("Residuals do NOT follow a normal distribution. Error term calculation is invalid.")
        residual_mean = None
        residual_stddev = None

    return {
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "e_s_mean": residual_mean,
        "e_s_stddev": residual_stddev
    }

# Example 
# coefficients = derive_recovery("mock_data_file.xlsx")
# print(coefficients)

# We need alt recovery method here which expects a recovery JUST BEFORE, and JUST AFTER tamping


def derive_defect_probabilities(file_path):
    """
    Derives parameters for predicting defect probabilities using ordinal logistic regression,
    using the explicit defect classification from the "defect" column.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Contains the threshold parameters (C0, C1), the coefficient for DLL_s (beta),
              and a goodness-of-fit metric.
    """
    import pandas as pd
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    import numpy as np

    # Load the Excel file
    df = pd.read_excel(file_path)

    # Check if either "DLL_s" or "A" exists
    if "DLL_s" not in df.columns and "A" not in df.columns:
        raise ValueError(f"Input file must contain either 'DLL_s' or 'A' as a column.")

    # If "A" is present but "DLL_s" is not, rename "A" to "DLL_s"
    if "DLL_s" not in df.columns and "A" in df.columns:
        df = df.rename(columns={"A": "DLL_s"})  # Rename internally

    # Ensure other necessary columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Prepare predictor and response variables.
    # X must be a DataFrame (even if it's a single column)
    X = df[["DLL_s"]]
    Y = df["Defect"]  # Ordinal response variable: expected values 0, 1, or 2

    # Initialize and fit the ordinal logistic regression model using the logit link.
    model = OrderedModel(Y, X, distr='logit')
    results = model.fit(method='bfgs', disp=False)

    # Extract parameter estimates.
    # The model returns threshold parameters (one less than the number of categories)
    # and the coefficient for DLL_s.
    params = results.params

    # Extract threshold parameters (they should be named something like 'threshold_1' and 'threshold_2')
    threshold_params = {k: v for k, v in params.items() if "threshold" in k}
    if len(threshold_params) != 2:
        raise ValueError("Expected exactly 2 threshold parameters for a 3-category ordinal response.")

    # Sort the threshold parameters by their keys to preserve the natural order.
    sorted_threshold_keys = sorted(threshold_params.keys())
    C0 = threshold_params[sorted_threshold_keys[0]]  # First threshold (separates category 0 and 1)
    C1 = threshold_params[sorted_threshold_keys[1]]  # Second threshold (separates category 1 and 2)

    # Extract the coefficient for DLL_s.
    if "DLL_s" not in params:
        raise ValueError("DLL_s coefficient not found in model parameters.")
    beta = params["DLL_s"]

    # Obtain a goodness-of-fit metric (pseudo R-squared in this case)
    gof = results.prsquared

    print("Derived Defect Probability Parameters:")
    print(f"C0 (Threshold between defect categories 0 and 1): {C0}")
    print(f"C1 (Threshold between defect categories 1 and 2): {C1}")
    print(f"beta (Coefficient for DLL_s): {beta}")
    print(f"Goodness-of-fit (Pseudo R-squared): {gof}")

    return {
        "C0": C0,
        "C1": C1,
        "beta": beta,
        "goodness_of_fit": gof
    }

# Example 
# coefficients = derive_defect_probabilities("mock_data_file.xlsx")
# print(coefficients)

def degrade(current_DLL_s, b_s, time, e_s):
    """
    Models the degradation progression over a time interval.
    

    Parameters:
        current_DLL_s (float): Current DLL_s value.
        b_s (float): Degradation rate.
        time (int): Number of days over which degradation occurs.
        e_s(float): Error term.

    """
    
    next_DLL_s = current_DLL_s + (b_s * time) + e_s
    return next_DLL_s
    

def recovery(current_DLL_s, recovery_coefficients, tamping_type, re_s_m, re_s_s):
    """
    Applies the recovery model to compute the updated DLL_s after tamping.
    re_s_m and re_s_s are the deciders for the error term
    
    Parameters:
    current_DLL_s (float): DLL_s before recovery. 
    recovery_coefficients (set of floats): Calculated recovery coefficients. 
    tamping_type (int): Binary 0 or 1, 0 for partial, 1 for complete.
    re_s_m (float): mean of recovery residuals
    re_s_s (float): standard deviation of recovery residuals

    """
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]

    epsilon_LL = np.random.normal(re_s_m, re_s_s)

    R_LL_s = (
        alpha_1 +
        beta_1 * current_DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * current_DLL_s +
        epsilon_LL
    )
    
    updated_DLL_s = current_DLL_s - R_LL_s
    return updated_DLL_s  

def defect(current_DLL_s, defect_coefficients):
    """
    Calculates defect probabilities.

    Parameters:
        current_DLL_s (float): Current DLL_s value.
        defect_coefficients (dict): Coefficients for the defect probability model.

    """
    C_0 = defect_coefficients["C_0"]
    C_1 = defect_coefficients["C_1"]
    b = defect_coefficients["b"]
    
    P_leq_1 = np.exp(C_0 + b * current_DLL_s) / (1 + np.exp(C_0 + b * current_DLL_s))
    P_leq_2 = np.exp(C_1 + b * current_DLL_s) / (1 + np.exp(C_1 + b * current_DLL_s))
    
    P1 = P_leq_1
    P2 = P_leq_2 - P_leq_1
    P3 = 1 - P1 - P2
    
    return {"P1": P1, "P2": P2, "P3": P3}



def sim_seg(time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, lognormal_mean, lognormal_stddev,e_s_mean, e_s_s, re_s_m, re_s_s, ILL, IALL, RM, RV):
    """
    Simulates a single segment of the track up to the specified time horizon.

    Parameters:
        time_horizon (int): Total simulation time in days.
        T_insp (int): Inspection interval in days.
        T_tamp (int): Regular tamping interval, in days
        T_step (int): Amount of time that passes for each step. Standard is 1. 
        AL (float): Alert limit for DLL_s.
        D_0LLs (float): Initial degradation value after tamping.
        lognormal_mean (float): Mean of the degradation rate.
        lognormal_stddev (float): Standard deviation of the degradation rate.
        e_s_s (float): Stddev of the Gaussian noise term for degradation.
        e_s_mean (float): mean of degradation error term (0.00 in test case)
        re_s_m (float): mean of recovery residuals.
        re_s_s (float): standard dev of recovery residuals.
        ILL (float): Limit for IL.
        IALL (float): Limit for IALL.
        RM (int): mean amount of days to perform tamping for IL defects.
        RV (int): Variance in days for time taken to tamp IL defects. 

    Returns:
        dict: Contains final state of simulation and maintenance counters.
    """
    # Var init
    t = 1  # Start time
    tn = 0  # Last tamping time
    Npm = 0  # Preventive maintenance counter
    Ncm_n = 0  # Corrective maintenance (normal) counter
    Ncm_e = 0  # Corrective maintenance (emergency) counter
    Ninsp = 0  # Inspection counter
    total_response_delay = 0 # total response delay accrued

    # b_s sampling
    b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)  # Sample degradation rate
    # print(f"Initial degradation rate b_s: {b_s}")

    while t <= time_horizon:
        # print(f"\n--- New Loop ---\nTime t: {t}, Last tamping time tn: {tn}")

        # Increment
        t += T_step
       #  print(f"Incremented time t: {t}")

        e_s = np.random.normal(e_s_mean, e_s_s)
       #  print(f"Sampled error term e_s: {e_s}")

        # Formula from paper
        DLL_s_t = D_0LLs + b_s * (t - tn) + e_s
        # print(f"Calculated DLL_s_t: {DLL_s_t}")
        # print (f"Original D0LL_s: {D_0LLs}")

        # Check if t is at tamping interval
        if t % T_tamp == 0:
            # print(f"Tamping interval reached at t = {t}")
            if DLL_s_t > AL:
               #  print(f"DLL_s_t ({DLL_s_t}) > AL ({AL}), performing preventive maintenance")
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 1,  re_s_m, re_s_s)
               #  print(f"Updated D_0LLs after recovery: {D_0LLs}")
                Npm += 1
                tn = t
                b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)
               #  print(f"Sampled new degradation rate b_s: {b_s}")
            else:
                print(f"DLL_s_t ({DLL_s_t}) <= AL ({AL}), no preventive maintenance needed")
            
            # protects against case the both conditions satisfy
            if t % T_insp == 0:
                print(f"Tamping interval coincides with inspection interval at t = {t}")
            else:    
                continue

        # Check if t is at an inspection interval
        if t % T_insp == 0:
            Ninsp += 1
            # print(f"Inspection interval reached at t = {t}")
            probabilities = defect(DLL_s_t, defect_coefficients)
            PIL = probabilities["P2"]
            PIAL = probabilities["P3"]
           #  print(f"Calculated defect probabilities: PIL = {PIL}, PIAL = {PIAL}")

            if PIAL > IALL:
               #  print(f"PIAL ({PIAL}) > IALL ({IALL}), performing emergency corrective maintenance")
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 0,  re_s_m, re_s_s)
               #  print(f"Updated D_0LLs after emergency recovery: {D_0LLs}")
                Ncm_e += 1
                tn = t
                continue

            elif PIL > ILL:
                # print(f"PIL ({PIL}) > ILL ({ILL}), performing normal corrective maintenance")
                time_to_next_inspection = T_insp - (t % T_insp)
                # print(f"Time to next inspection: {time_to_next_inspection}")

                response_time = min(max(0, np.random.normal(loc=RM, scale=RV)), time_to_next_inspection)
               #  print(f"Bounded response time: {response_time}")
                
                original_t = t

                t += response_time 
                response_time = int(response_time)
                
                total_response_delay += response_time
                
                t = int(t)
                DLL_s_t = degrade(DLL_s_t, b_s, response_time, e_s)
               #  print(f"Updated DLL_s_t after degradation: {DLL_s_t}")
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 0,  re_s_m, re_s_s)
                # print(f"Updated D_0LLs after normal recovery: {D_0LLs}")
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
    time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean, degradation_stddev,e_s_mean, e_s_variance,
    re_s_m, re_s_s, ILL, IALL, RM, RV, num_simulations,
    inspection_cost, preventive_maintenance_cost, normal_corrective_maintenance_cost,
    emergency_corrective_maintenance_cost
):
    """
    Monte Carlo simulation to aggregate results for multiple track sections.

    Parameters:
        Same as `sim_seg`, with additional parameters:
        - num_simulations (int): Number of simulations to run.
        - inspection_cost (float): Cost per inspection.
        - preventive_maintenance_cost (float): Cost of preventive maintenance.
        - normal_corrective_maintenance_cost (float): Cost of normal corrective maintenance.
        - emergency_corrective_maintenance_cost (float): Cost of emergency corrective maintenance.

    Returns:
        dict: Aggregated results from all simulations.
    """
    total_inspections = 0
    total_pm = 0
    total_cm_n = 0
    total_cm_e = 0
    total_cost = 0

    for i in range(num_simulations):
        result = sim_seg(
           time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean,
           degradation_stddev,e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV
       )

        # Get results from each sim
        inspections = result["Number of inspections"]
        pm = result["Npm"]
        cm_n = result["Ncm_n"]
        cm_e = result["Ncm_e"]

        # All costs
        cost = (
            inspections * inspection_cost +
            pm * preventive_maintenance_cost +
            cm_n * normal_corrective_maintenance_cost +
            cm_e * emergency_corrective_maintenance_cost
        )

        # aggregate
        total_inspections += inspections
        total_pm += pm
        total_cm_n += cm_n
        total_cm_e += cm_e
        total_cost += cost

    # all averages
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

def AL_analyze(
    time_horizon, T_insp, T_tamp, T_step, D_0LLs, degradation_mean,
    degradation_stddev, e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV,
    num_simulations, inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost,
    min_AL, max_AL, inc_AL, output_file="AL_analysis.xlsx"
):
    """
    Analyzes the performance of the Monte Carlo simulation over a range of AL values.

    Parameters:
        Same as `monte()`, with additional parameters:
        - min_AL (float): Starting value of AL.
        - max_AL (float): Ending value of AL.
        - inc_AL (float): Increment value for AL.
        - output_file (str): Name of the Excel file to save the results.

    Returns:
        None: Saves the results to an Excel file.
    """
    results = []
    trial_number = 1

    # Iterate over AL values
    current_AL = min_AL
    while current_AL <= max_AL:
        print(f"Running trial {trial_number} with AL = {current_AL:.2f}")

        # Run the Monte Carlo simulation with the current AL value
        result = monte(
            time_horizon, T_insp, T_tamp, T_step, current_AL, D_0LLs,
            degradation_mean, degradation_stddev, e_s_mean, e_s_variance,
            re_s_m, re_s_s, ILL, IALL, RM, RV, num_simulations,
            inspection_cost, preventive_maintenance_cost,
            normal_corrective_maintenance_cost,
            emergency_corrective_maintenance_cost
        )

        # Append results to the list
        results.append({
            "Trial Number": trial_number,
            "AL": current_AL,
            "ILL (IL)": ILL,  # Adding the constant ILL value
            "IALL (IAL)": IALL,  # Adding the constant IALL value
            "Mean Normal CM Actions": result["Average Normal Corrective Maintenances"],
            "Mean Emergency CM Actions": result["Average Emergency Corrective Maintenances"],
            "Mean PM Actions": result["Average Preventive Maintenances"]
        })

        # Increment AL and trial number
        current_AL += inc_AL
        trial_number += 1

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save to Excel
    df_results.to_excel(output_file, index=False)
    print(f"AL analysis results saved to {output_file}")


# The AL method given here is a sample method. I might have to build it downstream that we manipulate a target variable and see how it affects the pipeline. Think on this. I want it to be easy for the user
# but not make it so that I must write a method for each test case. It should be variable and adaptive

def AL_graph(
    time_horizon, T_insp, T_tamp, T_step, D_0LLs, degradation_mean,
    degradation_stddev, e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV,
    num_simulations, inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost,
    min_AL, max_AL, inc_AL, output_file="AL_analysis.xlsx"
):
    """
    Analyzes the performance of the Monte Carlo simulation over a range of AL values
    and generates a cost graph.

    Parameters:
        Same as `monte()`, with additional parameters:
        - min_AL (float): Starting value of AL.
        - max_AL (float): Ending value of AL.
        - inc_AL (float): Increment value for AL.
        - output_file (str): Name of the Excel file to save the results.

    Returns:
        None: Saves the results to an Excel file and plots the cost graph.
    """
    results = []
    trial_number = 1
    AL_values = []
    total_costs = []

    # Iterate over AL values
    current_AL = min_AL
    while current_AL <= max_AL:
        print(f"Running trial {trial_number} with AL = {current_AL:.2f}")

        # Run the Monte Carlo simulation with the current AL value
        result = monte(
            time_horizon, T_insp, T_tamp, T_step, current_AL, D_0LLs,
            degradation_mean, degradation_stddev, e_s_mean, e_s_variance,
            re_s_m, re_s_s, ILL, IALL, RM, RV, num_simulations,
            inspection_cost, preventive_maintenance_cost,
            normal_corrective_maintenance_cost,
            emergency_corrective_maintenance_cost
        )

        # Extract the total cost for the current AL
        total_cost = result["Average Cost"]

        # Append AL and cost to the lists for plotting
        AL_values.append(current_AL)
        total_costs.append(total_cost)

        # Append results to the list
        results.append({
            "Trial Number": trial_number,
            "AL": current_AL,
            "Total Cost": total_cost,
            "Mean Normal CM Actions": result["Average Normal Corrective Maintenances"],
            "Mean Emergency CM Actions": result["Average Emergency Corrective Maintenances"],
            "Mean PM Actions": result["Average Preventive Maintenances"]
        })

        # Increment AL and trial number
        current_AL += inc_AL
        trial_number += 1

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save to Excel
    df_results.to_excel(output_file, index=False)
    print(f"AL analysis results saved to {output_file}")

    # Plot the cost graph
    plt.figure(figsize=(10, 6))
    plt.plot(AL_values, total_costs, marker='o', linestyle='--', color='b', label="Total Maintenance Cost")
    plt.axhline(y=min(total_costs), color='r', linestyle='-', label="Minimum Cost")
    plt.xlabel("Maintenance Limit (AL, mm)")
    plt.ylabel("Total Maintenance Cost per Year (SEK)")
    plt.title("Effect of Maintenance Limit on Total Maintenance Cost")
    plt.legend()
    plt.grid(True)
    plt.show()


# Sim parameters (ideal)
time_horizon = 5 * 365  
T_insp = 4  
T_tamp = 12  
T_step = 1  
AL = 1.5  
ILL = 0.4 
IALL = 0.05  
RM = 35  
RV = 7  
degradation_mean = -2.379  
degradation_stddev = 0.756 
e_s_mean = 0
e_s_variance = 0.041 
re_s_m = 0
re_s_s = 0.15
D_0LLs = 0.352  


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

AL_analyze(
    time_horizon, T_insp, T_tamp, T_step, D_0LLs, degradation_mean,
    degradation_stddev, e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV,
    num_simulations, inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost,
    min_AL = 1.2 , max_AL = 1.95 , inc_AL = 0.05, output_file="AL_analysis.xlsx"
)




"""
PSI
"""

# TO DO: Fix PSI such that we do not use defect probabilities and only use the choices in the document - DONE, needs checked

# TO DO: Implement alternate b_s calc - DONE, but need to implement input framework based on empircal constants (cannot find)

# TO DO: Make it so that PSI can be calculated / extrapolated from raw weight load and selected class of track - NOT DONE, ask Sudipta

# TO DO: Gauge/Alignment Pipeline, basically the same, just different measurement - DONE For A, should be directly analagous (check), NOT DONE for G, it is usually thresholded eg
# 100 +- 20, so this means that G needs signficant adjustment (for example, does deg go down or up?), recovery should adjust TOWARDS the median, but we need decision for deg. Otherwise though
# the model is updated to expect A or DLLs, where A is the absolute measurement between BOTH sides. It is data driven so this should be ok. 

# TO DO: Verify all prior results
# TO DO: Verify all track measures
# TO DO: Input / Output formatting, help functions, etc for track measurements (Sweden Q, etc) 

# For meeting tmr: Notify that alternate b_s calc is finished, working on gauge and alignment, for sure done by Monday
# Apologize, midterms week, lot of coding
# We will be on track to finish all theoretical still by late March
# Inform him of my vision of roles. Past theoretical framework, I will move to solely testing and advising
# I will draft documents which James will build and beautify. I will directly oversee results 


def take_in_PSI(file_path):
    """
    Reads an Excel file and calculates constants and parameters related to PSI degradation and maintenance.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Calculated constants and parameters.
    """
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Ensure necessary columns exist
    required_columns = ["Date", "PSI", "Maintenance", "Maintenance Action"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Sort by date to ensure chronological order
    df = df.sort_values(by="Date")
    df["Date"] = pd.to_datetime(df["Date"])  # Ensure Date column is datetime

    # Calculate time intervals (in days)
    df["Time Interval"] = (df["Date"] - df["Date"].shift(1)).dt.days

    # Identify ranges for degradation rate calculations
    maintenance_indices = df[df["Maintenance"] == 1].index
    degradation_periods = []

    for i in range(len(maintenance_indices) - 1):
        start = maintenance_indices[i] + 1  # Start the day after maintenance
        end = maintenance_indices[i + 1]  # End on the next maintenance day
        degradation_periods.append(df.loc[start:end])

    # Calculate degradation rates for each period
    degradation_rates = []
    time_intervals = []

    for period in degradation_periods:
        period["Degradation Rate"] = -period["PSI"].diff() / period["Time Interval"]  # Negative since PSI decreases
        degradation_rates.extend(period["Degradation Rate"].dropna().tolist())
        time_intervals.extend(period["Time Interval"].dropna().tolist())

    # Test for lognormal distribution (on degradation rates, since PSI itself is decreasing)
    log_psi_values = np.log(-df["PSI"].diff().dropna())  # We take negative diff to get degradation as positive values
    result = anderson(log_psi_values, dist="norm")

    if result.significance_level[0] > 0.05:
        print("PSI degradation rates follow a lognormal distribution.")
        lognormal_mean = np.mean(log_psi_values)
        lognormal_stddev = np.std(log_psi_values)

        degradation_mean = np.exp(lognormal_mean + 0.5 * lognormal_stddev**2)
        degradation_stddev = np.sqrt((np.exp(lognormal_stddev**2) - 1) * np.exp(2 * lognormal_mean + lognormal_stddev**2))
    else:
        print("PSI degradation rates do not follow a lognormal distribution. Assuming normal distribution.")
        degradation_mean = np.mean(-df["PSI"].diff().dropna())  # Taking negative since PSI decreases
        degradation_stddev = np.std(-df["PSI"].diff().dropna())

    print(f"Degradation Mean: {degradation_mean}")
    print(f"Degradation Stddev: {degradation_stddev}")

    # Extract the most recent maintenance event to find initial PSI post-maintenance
    most_recent_maintenance = df[df["Maintenance"] == 1].iloc[-1]
    most_recent_index = most_recent_maintenance.name

    if most_recent_index + 1 < len(df):
        PSI_0 = df.iloc[most_recent_index + 1]["PSI"]
    else:
        PSI_0 = most_recent_maintenance["PSI"]

    print(f"PSI_0 (Initial PSI value after last maintenance): {PSI_0}")

    return {
        "PSI_0": PSI_0,
        "degradation_mean": degradation_mean,
        "degradation_stddev": degradation_stddev,
        "degradation_rates": degradation_rates,
        "time_intervals": time_intervals,
    }


def degrade_PSI(current_PSI, b_s, time, e_s):
    """
    Models PSI degradation over a given time interval.

    Parameters:
        current_PSI (float): Current PSI value.
        b_s (float): Degradation rate (lognormal-based).
        time (int): Number of days over which degradation occurs.
        e_s (float): Error term (Gaussian noise).

    Returns:
        float: Updated PSI value after degradation.
    """
    next_PSI = current_PSI - (b_s * time) - e_s  # PSI decreases over time
    return max(next_PSI, 0)  # Ensures PSI does not drop below 0


def recovery_PSI(current_PSI, recovery_coefficients, maintenance_action, re_s_m, re_s_s):
    """
    Applies the recovery model to compute the updated PSI after maintenance.

    Parameters:
        current_PSI (float): PSI before recovery.
        recovery_coefficients (dict): Calculated recovery coefficients.
        maintenance_action (int): Maintenance class (1=Minor, 2=Moderate, 3=Major, 4=Full Reset).
        re_s_m (float): Mean of recovery residuals.
        re_s_s (float): Standard deviation of recovery residuals.

    Returns:
        float: Updated PSI after recovery, or 1.0 if a full reset occurs.
    """

    # If Maintenance Action is 4, PSI resets to 1.0
    if maintenance_action == 4:
        return 1.0  # Special case handled in loop logic

    # Map maintenance action levels
    if maintenance_action == 1:
        tamping_effect = 0.0  # Minor improvement
    elif maintenance_action == 2:
        tamping_effect = 1.0  # Moderate improvement
    elif maintenance_action == 3:
        tamping_effect = 1.5  # Major improvement
    else:
        raise ValueError(f"Invalid Maintenance Action: {maintenance_action}")

    # Extract regression coefficients
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]

    # Sample random error term from normal distribution
    epsilon_PSI = np.random.normal(re_s_m, re_s_s)

    # Compute recovery effect
    R_PSI = (
        alpha_1 +
        beta_1 * current_PSI +
        beta_2 * tamping_effect +
        beta_3 * tamping_effect * current_PSI +
        epsilon_PSI
    )

    # PSI increases after maintenance
    updated_PSI = current_PSI + R_PSI
    return min(updated_PSI, 1.0)  # Ensure PSI does not exceed 1.0



def derive_recovery_PSI(file_path):
    """
    Derives recovery coefficients for PSI-based track maintenance using regression and validates residual normality.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Derived recovery coefficients {alpha_1, beta_1, beta_2, beta_3} and error term stats {mean, stddev}.
    """
    df = pd.read_excel(file_path)

    # Ensure necessary columns exist
    required_columns = ["Date", "PSI", "Maintenance", "Maintenance Action", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Filter for maintenance events
    maintenance_events = df[df["Maintenance"] == 1]

    # Ensure we have enough data points
    if len(maintenance_events) < 4:
        raise ValueError("Not enough maintenance events to perform regression.")

    # Calculate recovery (ΔPSI = PSI_after - PSI_before)
    R_PSI = []
    PSI_values = df["PSI"].values

    for index in maintenance_events.index:
        if index + 1 < len(df):
            R_PSI.append(PSI_values[index + 1] - PSI_values[index])  # Now adding instead of subtracting

    # Drop last maintenance event if no pair exists
    maintenance_events = maintenance_events.iloc[:-1]

    # Prepare regression input
    X = maintenance_events[["PSI", "Maintenance Action"]].copy()
    X["Maintenance Action"] = X["Maintenance Action"].map({1: 0, 2: 1, 3: 1.5})  # Minor = 0, Moderate = 1, Major = 1.5
    X["Interaction"] = X["PSI"] * X["Maintenance Action"]  # Interaction term

    X = X.values
    Y = np.array(R_PSI)

    # Perform regression
    model = LinearRegression()
    model.fit(X, Y)

    # Extract coefficients
    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    print(f"Derived PSI Recovery Coefficients:")
    print(f"alpha_1 (Intercept): {alpha_1}")
    print(f"beta_1 (Coefficient for PSI): {beta_1}")
    print(f"beta_2 (Coefficient for Maintenance Action): {beta_2}")
    print(f"beta_3 (Interaction Coefficient): {beta_3}")

    # Calculate residuals
    residuals = Y - model.predict(X)

    # Test residuals for normality
    ad_test_result = anderson(residuals, dist='norm')

    if ad_test_result.significance_level[2] > 0.05:
        print("Residuals follow a normal distribution.")
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        print(f"Error Term Mean (e_s_mean): {residual_mean}")
        print(f"Error Term Stddev (e_s_stddev): {residual_stddev}")
    else:
        print("Residuals do NOT follow a normal distribution. Error term calculation is invalid.")
        residual_mean = None
        residual_stddev = None

    return {
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "e_s_mean": residual_mean,
        "e_s_stddev": residual_stddev
    }


def derive_accurate_recovery_PSI(file_path, bs_values):
    """
    Derives recovery coefficients from the PSI dataset using regression,
    adjusting the observed recovery effect by subtracting expected degradation
    over the recovery period.

    Parameters:
        file_path (str): Path to the Excel file containing track data.
        bs_values (array-like): Array of daily PSI degradation rates.

    Returns:
        dict: Derived recovery coefficients {alpha_1, beta_1, beta_2, beta_3} and error term stats {mean, stddev}.
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import anderson

    # Load the dataset
    df = pd.read_excel(file_path)

    # Ensure necessary columns exist
    required_columns = ["Date", "PSI", "Maintenance", "Maintenance Action", "Defect"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter for maintenance events
    maintenance_events = df[df["Maintenance"] == 1]
    if len(maintenance_events) < 4:
        raise ValueError("Not enough maintenance events to perform regression.")

    # Compute the average daily degradation rate from bs_values
    average_bs = np.mean(bs_values)

    # Compute the adjusted recovery effect (R_PSI_adjusted) for each maintenance pair.
    PSI_values = df["PSI"].values
    R_PSI_adjusted = []
    maintenance_indices = maintenance_events.index.tolist()

    for i in range(len(maintenance_indices) - 1):
        idx = maintenance_indices[i]
        next_idx = maintenance_indices[i + 1]

        # Observed recovery effect (increase in PSI)
        observed_R = PSI_values[next_idx] - PSI_values[idx]  # Recovery increases PSI

        # Determine the time interval (in days) between these two measurements
        t1 = df.loc[idx, "Date"]
        t2 = df.loc[next_idx, "Date"]
        interval = (t2 - t1).days

        # Adjust the recovery effect by subtracting expected degradation
        adjusted_R = observed_R - (average_bs * interval)
        R_PSI_adjusted.append(adjusted_R)

    # Use maintenance events (except the last one) for regression.
    maintenance_events = maintenance_events.iloc[:-1]

    # Prepare regression input
    X = maintenance_events[["PSI", "Maintenance Action"]].copy()
    X["Maintenance Action"] = X["Maintenance Action"].map({1: 0, 2: 1, 3: 1.5})  # Minor = 0, Moderate = 1, Major = 1.5
    X["Interaction"] = X["PSI"] * X["Maintenance Action"]  # Interaction term

    X = X.values
    Y = np.array(R_PSI_adjusted)

    # Perform regression
    model = LinearRegression()
    model.fit(X, Y)

    # Extract regression coefficients
    alpha_1 = model.intercept_
    beta_1, beta_2, beta_3 = model.coef_

    print("Derived Accurate PSI Recovery Coefficients:")
    print(f"alpha_1 (Intercept): {alpha_1}")
    print(f"beta_1 (Coefficient for PSI): {beta_1}")
    print(f"beta_2 (Coefficient for Maintenance Action): {beta_2}")
    print(f"beta_3 (Interaction Coefficient): {beta_3}")

    # Compute residuals and test for normality
    residuals = Y - model.predict(X)
    ad_test_result = anderson(residuals, dist='norm')

    if ad_test_result.significance_level[2] > 0.05:
        print("Residuals follow a normal distribution.")
        residual_mean = np.mean(residuals)
        residual_stddev = np.std(residuals)
        print(f"Error Term Mean (e_s_mean): {residual_mean}")
        print(f"Error Term Stddev (e_s_stddev): {residual_stddev}")
    else:
        print("Residuals do NOT follow a normal distribution. Error term calculation is invalid.")
        residual_mean = None
        residual_stddev = None

    return {
        "alpha_1": alpha_1,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "e_s_mean": residual_mean,
        "e_s_stddev": residual_stddev
    }


def sim_seg_PSI(
    time_horizon, T_insp, T_tamp, T_step, AL, PSI_0, lognormal_mean, lognormal_stddev,
    e_s_mean, e_s_s, re_s_m, re_s_s, RM, RV
):
    """
    Simulates a single segment of the track up to the specified time horizon, 
    using PSI-based degradation and maintenance decisions.

    Parameters:
        time_horizon (int): Total simulation time in days.
        T_insp (int): Inspection interval in days.
        T_tamp (int): Regular tamping interval, in days.
        T_step (int): Time step increment (default is 1 day).
        AL (float): Alert limit for PSI.
        PSI_0 (float): Initial PSI value after last maintenance.
        lognormal_mean (float): Mean of the degradation rate.
        lognormal_stddev (float): Standard deviation of the degradation rate.
        e_s_mean (float): Mean of degradation error term.
        e_s_s (float): Stddev of Gaussian noise for degradation.
        re_s_m (float): Mean of recovery residuals.
        re_s_s (float): Standard deviation of recovery residuals.
        RM (int): Mean days to perform maintenance for planned repairs.
        RV (int): Variance in days for maintenance delays.

    Returns:
        dict: Contains final state of simulation and maintenance counters.
    """
    # Initialize variables
    t = 1  # Start time
    tn = 0  # Last maintenance time
    Npm = 0  # Preventive maintenance counter
    Ncm_n = 0  # Corrective maintenance (normal) counter
    Ncm_e = 0  # Corrective maintenance (emergency) counter
    Ninsp = 0  # Inspection counter
    total_response_delay = 0  # Total response delay accrued
    PSI_t = PSI_0  # Current PSI state

    # Sample initial degradation rate
    b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)

    while t <= time_horizon:
        t += T_step  # Increment time step

        # Apply degradation
        e_s = np.random.normal(e_s_mean, e_s_s)
        PSI_t = max(PSI_t - (b_s * T_step) - e_s, 0)  # PSI degrades over time

        # Preventive maintenance at tamping intervals
        if t % T_tamp == 0:
            if PSI_t <= AL:
                # Preventive maintenance performed if PSI has degraded past threshold
                PSI_t = recovery_PSI(PSI_t, {"alpha_1": re_s_m, "beta_1": 0, "beta_2": 0, "beta_3": 0}, 1, re_s_m, re_s_s)
                Npm += 1
                tn = t
                b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)

            if t % T_insp == 0:
                continue  # Skip duplicate processing when tamping coincides with inspection

        # Inspection intervals
        if t % T_insp == 0:
            Ninsp += 1

            # Maintenance decision based on PSI thresholds
            if 0.6 < PSI_t <= 1.0:
                continue  # No action, only monitoring

            elif 0.4 < PSI_t <= 0.6:
                # Partial replacement (Moderate Maintenance)
                PSI_t = recovery_PSI(PSI_t, {"alpha_1": re_s_m, "beta_1": 0, "beta_2": 1, "beta_3": 0}, 2, re_s_m, re_s_s)
                Ncm_n += 1
                tn = t

            elif 0.2 < PSI_t <= 0.4:
                # Planning of renewal (Major Maintenance)
                PSI_t = recovery_PSI(PSI_t, {"alpha_1": re_s_m, "beta_1": 0, "beta_2": 1.5, "beta_3": 0}, 3, re_s_m, re_s_s)
                Ncm_n += 1
                tn = t

            elif 0.0 <= PSI_t <= 0.2:
                # Full renewal (Emergency Maintenance)
                PSI_t = 1.0  # Complete reset
                Ncm_e += 1
                tn = t

    return {
        "final_t": t,
        "final_PSI": PSI_t,
        "b_s": b_s,
        "Npm": Npm,
        "Ncm_n": Ncm_n,
        "Ncm_e": Ncm_e,
        "Total Response Delay": total_response_delay,
        "Number of inspections": Ninsp
    }


def monte_PSI(
    time_horizon, T_insp, T_tamp, T_step, AL, PSI_0, degradation_mean, degradation_stddev, 
    e_s_mean, e_s_variance, re_s_m, re_s_s, RM, RV, num_simulations, 
    inspection_cost, preventive_maintenance_cost, normal_corrective_maintenance_cost, 
    emergency_corrective_maintenance_cost
):
    """
    Monte Carlo simulation to aggregate results for multiple track sections, using PSI-based maintenance decisions.

    Parameters:
        Same as `sim_seg_PSI`, with additional parameters:
        - num_simulations (int): Number of simulations to run.
        - inspection_cost (float): Cost per inspection.
        - preventive_maintenance_cost (float): Cost of preventive maintenance.
        - normal_corrective_maintenance_cost (float): Cost of normal corrective maintenance.
        - emergency_corrective_maintenance_cost (float): Cost of emergency corrective maintenance.

    Returns:
        dict: Aggregated results from all simulations.
    """
    total_inspections = 0
    total_pm = 0
    total_cm_n = 0
    total_cm_e = 0
    total_cost = 0

    for _ in range(num_simulations):
        result = sim_seg_PSI(
            time_horizon, T_insp, T_tamp, T_step, AL, PSI_0, degradation_mean, 
            degradation_stddev, e_s_mean, e_s_variance, re_s_m, re_s_s, RM, RV
        )

        # Extract results
        inspections = result["Number of inspections"]
        pm = result["Npm"]
        cm_n = result["Ncm_n"]
        cm_e = result["Ncm_e"]

        # Calculate total cost
        cost = (
            inspections * inspection_cost +
            pm * preventive_maintenance_cost +
            cm_n * normal_corrective_maintenance_cost +
            cm_e * emergency_corrective_maintenance_cost
        )

        # Aggregate results
        total_inspections += inspections
        total_pm += pm
        total_cm_n += cm_n
        total_cm_e += cm_e
        total_cost += cost

    # Compute averages
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


def alt_bs(x, gamma, alpha, beta):
    # Defined in terms of axle rotations OR tonnage (x), y is determined
    # Important that the constants (alpha, beta, gamma) are calibrated in regards to rotation or tonnage, as well as for SDLLs SPECIFICALLY
    # This should be an accurate guesstimate though. The data-driven method should be better though
    
    # WHEN I CAN FIND ACCURATE constants such that a, b, g is described in terms of SDLLs and x, then this should be easy to present to user as table
    # I do not have access to article that claims to have this empirical information though 

    # Calculate degradation using the empirical formula
    y = gamma * (1 - np.exp(-alpha * x)) + beta * x
    
    return y








