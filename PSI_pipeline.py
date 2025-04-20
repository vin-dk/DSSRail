# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import anderson
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""
PSI
"""
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


def calculate_psi(cumulative_load, kf=KF, ml=ML):
    # given constant
    kf = 5.2
    return 1 - np.exp(kf * ((cumulative_load / ml) - 1))


def determine_psi(input_file, traffic_load_per_year, material_type, output_file="PSI_conver.xlsx"):
    """
    Reads an Excel file, calculates PSI values based on cumulative load, and saves the transformed data.

    :param input_file: Path to the input Excel file
    :param traffic_load_per_year: Annual traffic load (MGT per year)
    :param material_type: Material type (1-6)
    :param output_file: Path to save the output Excel file
    """
    # material constants
    AGE_LIMITS = {1: 45, 2: 40, 3: 25, 4: 30, 5: 18, 6: 21}    
    
    if material_type not in AGE_LIMITS:
        raise ValueError("Invalid material type. Must be one of: 1, 2, 3, 4, 5, 6")

    ML = AGE_LIMITS[material_type] * traffic_load_per_year

    df = pd.read_excel(input_file)

    df["PSI"] = df["Cumulative Load"].apply(lambda cl: calculate_psi(cl, ml=ML))

    output_df = df[["Date", "PSI", "Maintenance", "Maintenance Action"]]

    output_df.to_excel(output_file, index=False)

    print(f"Processed file saved as {output_file}")