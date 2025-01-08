# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import anderson
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LinearRegression

"""
THESE ARE THE PARAMETERS FOR A GOOD RUN
"""

# Constants and parameters
D_0LLs = 0.352  # Initial degradation value after tamping (given in the paper)
degradation_mean = -2.379  # Log mean for degradation rate
degradation_stddev = 0.756  # Log standard deviation for degradation rate
e_s_mean = 0  # Mean of the Gaussian error term
e_s_variance = 0.041  # Variance of the Gaussian error term

# Parameters for recovery model
recovery_coefficients = {
    "alpha_1": -0.269,
    "beta_1": 0.51,
    "beta_2": 0.207,
    "beta_3": -0.043
}
e_LL_mean = 0  # Mean of the Gaussian noise for recovery
e_LL_variance = 0.15  # Variance of the Gaussian noise for recovery

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

    # Ensure necessary columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Sort by date to ensure chronological order
    df = df.sort_values(by="Date")

    # Determine D_0LLs
    most_recent_tamping = df[df["Tamping Performed"] == 1].iloc[-1]
    most_recent_index = most_recent_tamping.name
    
    if most_recent_index + 1 < len(df):
        D_0LLs = df.iloc[most_recent_index + 1]["DLL_s"]
    else:
        D_0LLs = most_recent_tamping["DLL_s"]

    print(f"D_0LLs (Initial degradation value after tamping): {D_0LLs}")

    # Extract DLL_s values
    dll_values = df["DLL_s"].values

    # Test for lognormal distribution
    log_dll_values = np.log(dll_values[dll_values > 0])  # Log-transform positive values only
    result = anderson(log_dll_values, dist='norm')

    if result.significance_level[0] > 0.05:  # Assuming 5% significance level
        print("DLL_s values follow a lognormal distribution.")
        
        μ = np.mean(log_dll_values)  # Mean of log-transformed values
        σ = np.std(log_dll_values)  # Stddev of log-transformed values

        # Convert back to lognormal scale
        degradation_mean = np.exp(μ + 0.5 * σ**2)
        degradation_stddev = np.sqrt((np.exp(σ**2) - 1) * np.exp(2 * μ + σ**2))
    else:
        print("DLL_s values do not follow a lognormal distribution. Assuming normal distribution.")
        degradation_mean = np.mean(dll_values)
        degradation_stddev = np.std(dll_values)

    print(f"Degradation Mean: {degradation_mean}")
    print(f"Degradation Stddev: {degradation_stddev}")

    # Calculate error term variance
    residuals = dll_values - (D_0LLs + degradation_mean)
    e_s_variance = np.var(residuals)
    e_s_mean = np.mean(residuals)  # Should be near zero

    print(f"Error Term Mean (e_s_mean): {e_s_mean}")
    print(f"Error Term Variance (e_s_variance): {e_s_variance}")

    # Calculate Recovery Values
    recovery_coefficients = {
        "alpha_1": -0.269,
        "beta_1": 0.51,
        "beta_2": 0.207,
        "beta_3": -0.043
    }

    # Map tamping types: 2 -> 1 (complete), 1 -> 0 (partial)
    df["x_s"] = df["Tamping Type"].map({2: 1, 1: 0}).fillna(0)

    recovery_values = []
    for _, row in df.iterrows():
        if row["Tamping Performed"] == 1:
            DLL_s = row["DLL_s"]
            x_s = row["x_s"]
            epsilon_LL = np.random.normal(0, np.sqrt(0.15))

            R_LL_s = (
                recovery_coefficients["alpha_1"] +
                recovery_coefficients["beta_1"] * DLL_s +
                recovery_coefficients["beta_2"] * x_s +
                recovery_coefficients["beta_3"] * x_s * DLL_s +
                epsilon_LL
            )
            recovery_values.append(R_LL_s)
        else:
            recovery_values.append(None)

    df["Recovery Value"] = recovery_values

    # Print recovery values for debugging
    print("Recovery values calculated:")
    print(df[["Date", "DLL_s", "Tamping Performed", "Tamping Type", "Recovery Value"]])

    # Return calculated constants and parameters
    return {
        "D_0LLs": D_0LLs,
        "degradation_mean": degradation_mean,
        "degradation_stddev": degradation_stddev,
        "e_s_mean": e_s_mean,
        "e_s_variance": e_s_variance,
        "recovery_values": df["Recovery Value"].tolist()
    }

def derive_recovery(file_path):
    # NOTE, the inherent degradation, by nature of observation periods, are included in RLL_s calculation
    # paper does not mention???
    """
    Derives recovery coefficients from the provided dataset using regression and validates residual normality.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Derived recovery coefficients {alpha_1, beta_1, beta_2, beta_3} and error term stats {mean, stddev}.
    """
    # Read Excel file
    df = pd.read_excel(file_path)

    # Ensure necessary columns exist
    required_columns = ["Date", "DLL_s", "Tamping Performed", "Tamping Type"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Filter tamping rows
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

    tamping_events = tamping_events.iloc[:-1]  # Exclude last tamping event if no subsequent value exists

    # Prepare regression input
    X = tamping_events[["DLL_s", "Tamping Type"]].copy()
    X["Tamping Type"] = X["Tamping Type"].map({1: 0, 2: 1})  # Map tamping type: 1 -> 0 (partial), 2 -> 1 (complete)
    X["Interaction"] = X["DLL_s"] * X["Tamping Type"]  # Interaction term

    # Convert to numpy arrays
    X = X.values
    Y = np.array(R_LL_s)

    # Perform regression
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

    # Test residuals for normality using Anderson-Darling test
    ad_test_result = anderson(residuals, dist='norm')

    if ad_test_result.significance_level[2] > 0.05:  # Check p-value (5% significance level)
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

# Example 
# coefficients = derive_recovery("mock_data_file.xlsx")
# print(coefficients)


def derive_defect_probabilities(file_path):
    """
    Derives parameters for predicting defect probabilities using ordinal logistic regression.

    Parameters:
        file_path (str): Path to the Excel file containing track data.

    Returns:
        dict: Coefficients {C0, C1, beta} and goodness-of-fit metrics.
    """
    # Load data
    df = pd.read_excel(file_path)

    # Ensure necessary columns exist
    required_columns = ["DLL_s"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    # Assign defect levels based on thresholds
    def assign_defect_level(dll_s):
        if dll_s < 1.5:
            return 1  # No defect
        elif dll_s < 3.0:
            return 2  # Level A defect
        else:
            return 3  # Level B defect

    df["Defect Level"] = df["DLL_s"].apply(assign_defect_level)

    # Extract relevant data
    X = df["DLL_s"].values
    Y = df["Defect Level"].values  # 1: No defect, 2: Level A defect, 3: Level B defect

    # Fit ordinal logistic regression
    model = OrderedModel(Y, X, distr='logit')
    results = model.fit()

    # Extract coefficients
    params = results.params
    C0 = params[0]  # Intercept for transition from Level 1 to Level 2
    C1 = params[1]  # Intercept for transition from Level 2 to Level 3
    beta = params[2]  # Coefficient for DLL_s

    # Goodness-of-fit
    gof = results.prsquared  # McFadden's R-squared

    print("Derived Defect Probability Parameters:")
    print(f"C0 (Intercept for no defect): {C0}")
    print(f"C1 (Intercept for level A defect): {C1}")
    print(f"Beta (Coefficient for DLL_s): {beta}")
    print(f"Goodness-of-fit: {gof}")

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
    Models the degradation progression over a given time interval.
    
    The intention is to call this daily. EG t = time since tamped.

    Parameters:
        current_DLL_s (float): Current DLL_s value.
        degradation_rate_mean (float): Log mean of degradation rate.
        degradation_rate_stddev (float): Log standard deviation of degradation rate.
        time (int): Number of days over which degradation occurs.

    Returns:
        float: Updated DLL_s value after degradation.
    """
    
    next_DLL_s = current_DLL_s + (b_s * time) + e_s
    return next_DLL_s
    

def recovery(current_DLL_s, recovery_coefficients, tamping_type, re_s_m, re_s_s):
    """
    Applies the recovery model to compute the updated DLL_s after tamping.
    re_s_m and re_s_s are the deciders for the error term

    Returns:
        float: Updated DLL_s value after recovery.
    """
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]

    noise_stddev = np.sqrt(re_s_s)
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
    Calculates defect probabilities using ordinal logistic regression.

    Parameters:
        current_DLL_s (float): Current DLL_s value.
        defect_coefficients (dict): Coefficients for the defect probability model.

    Returns:
        dict: Probabilities for defect levels (P1, P2, P3).
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



def sim_seg(time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean, degradation_stddev,e_s_mean, e_s_variance, re_s_m, re_s_s, ILL, IALL, RM, RV):
    """
    Simulates a single segment of the track up to the specified time horizon.

    Parameters:
        time_horizon (int): Total simulation time in days.
        T_insp (int): Inspection interval in days.
        T_tamp (int): Regular tamping interval, in days
        T_step (int): Amount of time that passes for each step. Standard is 1. 
        AL (float): Alert limit for DLL_s.
        D_0LLs (float): Initial degradation value after tamping.
        degradation_mean (float): Mean of the degradation rate (lognormal scale).
        degradation_stddev (float): Standard deviation of the degradation rate (lognormal scale).
        e_s_variance (float): Variance of the Gaussian noise term for degradation.
        e_s_mean (float): mean of degradation error term (0.00 in test case)
        re_s_m (float): mean of recovery residuals
        re_s_v (float): standard dev of recovery residuals
        ILL (float): Limit for IL
        IALL (float): Limit for IALL
        RM (int): mean amount of days to perform tamping for IL defects
        RV (int): Variance in days for time taken to tamp IL defects 

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
    b_s = np.random.lognormal(mean=degradation_mean, sigma=degradation_stddev)  # Sample degradation rate
    # print(f"Initial degradation rate b_s: {b_s}")

    while t <= time_horizon:
        # print(f"\n--- New Loop ---\nTime t: {t}, Last tamping time tn: {tn}")

        # Increment
        t += T_step
       #  print(f"Incremented time t: {t}")

        e_s = np.random.normal(e_s_mean, np.sqrt(e_s_variance))
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
                b_s = np.random.lognormal(mean=degradation_mean, sigma=degradation_stddev)
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


# Sim parameters (ideal)
time_horizon = 5 * 365  
T_insp = 32  
T_tamp = 15  
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
    min_AL = 1.2 , max_AL = 1.9 , inc_AL = 0.05, output_file="AL_analysis.xlsx"
)