# Import necessary libraries
import numpy as np
import pandas as pd

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
num_simulations = 80000  # Total number of simulation runs
inspection_interval = 120  # Time interval between inspections (days)
simulation_time_horizon = 15 * 365  # Total simulation time horizon (15 years)

# Tamping parameters
response_time_mean = 5 * 7  # Mean response time in days (5 weeks)
response_time_stddev = 7  # Standard deviation for response time (1 week)

# Derived constants
time_step = 1  # Daily degradation update


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
    
    # Apply degradation
    next_DLL_s = current_DLL_s + (b_s * time) + e_s
    return next_DLL_s
    

def recovery(current_DLL_s, recovery_coefficients, tamping_type):
    """
    Applies the recovery model to compute the updated DLL_s after tamping.

    Returns:
        float: Updated DLL_s value after recovery.
    """
    # Extract coefficients
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]

    # Add noise to the recovery process
    noise_stddev = np.sqrt(0.15)
    epsilon_LL = np.random.normal(0, noise_stddev)

    # Compute recovery value
    R_LL_s = (
        alpha_1 +
        beta_1 * current_DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * current_DLL_s +
        epsilon_LL
    )
    
    # Apply recovery to current DLL_s
    updated_DLL_s = current_DLL_s - R_LL_s
    return updated_DLL_s  # Add return statement

def defect(current_DLL_s, defect_coefficients):
    """
    Calculates defect probabilities using ordinal logistic regression.

    Parameters:
        current_DLL_s (float): Current DLL_s value.
        defect_coefficients (dict): Coefficients for the defect probability model.

    Returns:
        dict: Probabilities for defect levels (P1, P2, P3).
    """
    # Extract coefficients
    C_0 = defect_coefficients["C_0"]
    C_1 = defect_coefficients["C_1"]
    b = defect_coefficients["b"]
    
    # Calculate cumulative probabilities
    P_leq_1 = np.exp(C_0 + b * current_DLL_s) / (1 + np.exp(C_0 + b * current_DLL_s))
    P_leq_2 = np.exp(C_1 + b * current_DLL_s) / (1 + np.exp(C_1 + b * current_DLL_s))
    
    # Convert to individual probabilities
    P1 = P_leq_1
    P2 = P_leq_2 - P_leq_1
    P3 = 1 - P1 - P2
    
    return {"P1": P1, "P2": P2, "P3": P3}



def sim_seg(time_horizon, T_insp, T_tamp, T_step, AL, D_0LLs, degradation_mean, degradation_stddev, e_s_variance, ILL, IALL, RM, RV):
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
        ILL (float): Limit for IL
        IALL (float): Limit for IALL
        RM (int): mean amount of days to perform tamping for IL defects
        RV (int): Variance in days for time taken to tamp IL defects 

    Returns:
        dict: Contains final state of simulation and maintenance counters.
    """
    # Initialization of variables
    t = 1  # Start time
    tn = 0  # Last tamping time
    Npm = 0  # Preventive maintenance counter
    Ncm_n = 0  # Corrective maintenance (normal) counter
    Ncm_e = 0  # Corrective maintenance (emergency) counter

    # Degradation model parameters
    b_s = np.random.lognormal(mean=degradation_mean, sigma=degradation_stddev)  # Sample degradation rate
    print(f"Initial degradation rate b_s: {b_s}")

    while t <= time_horizon:
        print(f"\n--- New Loop ---\nTime t: {t}, Last tamping time tn: {tn}")

        # Increment time
        t += T_step
        print(f"Incremented time t: {t}")

        # Sample error term
        e_s = np.random.normal(0, np.sqrt(e_s_variance))
        print(f"Sampled error term e_s: {e_s}")

        # Calculate DLL_s(t) using the formula
        DLL_s_t = D_0LLs + b_s * (t - tn) + e_s
        print(f"Calculated DLL_s_t: {DLL_s_t}")
        print (f"Original D0LL_s: {D_0LLs}")

        # Check if t is at tamping interval
        if t % T_tamp == 0:
            print(f"Tamping interval reached at t = {t}")
            if DLL_s_t > AL:
                print(f"DLL_s_t ({DLL_s_t}) > AL ({AL}), performing preventive maintenance")
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 1)
                print(f"Updated D_0LLs after recovery: {D_0LLs}")
                Npm += 1
                tn = t
                b_s = np.random.lognormal(mean=degradation_mean, sigma=degradation_stddev)
                print(f"Sampled new degradation rate b_s: {b_s}")
            else:
                print(f"DLL_s_t ({DLL_s_t}) <= AL ({AL}), no preventive maintenance needed")
                
            if t % T_insp == 0:
                print(f"Tamping interval coincides with inspection interval at t = {t}")
            else:    
                continue

        # Check if t is at an inspection interval
        if t % T_insp == 0:
            print(f"Inspection interval reached at t = {t}")
            probabilities = defect(DLL_s_t, defect_coefficients)
            PIL = probabilities["P2"]
            PIAL = probabilities["P3"]
            print(f"Calculated defect probabilities: PIL = {PIL}, PIAL = {PIAL}")

            if PIAL > IALL:
                print(f"PIAL ({PIAL}) > IALL ({IALL}), performing emergency corrective maintenance")
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 0)
                print(f"Updated D_0LLs after emergency recovery: {D_0LLs}")
                Ncm_e += 1
                tn = t
                continue

            elif PIL > ILL:
                print(f"PIL ({PIL}) > ILL ({ILL}), performing normal corrective maintenance")
                # Calculate remaining time until the next inspection
                time_to_next_inspection = T_insp - (t % T_insp)
                print(f"Time to next inspection: {time_to_next_inspection}")

                # Bound response time to the remaining time until the next inspection
                response_time = min(max(0, np.random.normal(loc=RM, scale=RV)), time_to_next_inspection)
                print(f"Bounded response time: {response_time}")

                t += response_time  # Increment time
                DLL_s_t = degrade(DLL_s_t, b_s, response_time, e_s)
                print(f"Updated DLL_s_t after degradation: {DLL_s_t}")
                D_0LLs = recovery(DLL_s_t, recovery_coefficients, 0)
                print(f"Updated D_0LLs after normal recovery: {D_0LLs}")
                Ncm_n += 1
                tn = t
                continue

    return {
        "final_t": t,
        "final_DLL_s": DLL_s_t,
        "b_s": b_s,
        "Npm": Npm,
        "Ncm_n": Ncm_n,
        "Ncm_e": Ncm_e
    }


# Simulation parameters
time_horizon = 5 * 365  # 5 years converted to days
T_insp = 32  # Inspection interval in days
T_tamp = 15  # Tamping interval in days
T_step = 1  # Daily time steps
AL = 1.5  # Alert limit
ILL = 0.75 # IL limit
IALL = 0.05  # IAL limit
RM = 35  # Mean response time for corrective maintenance
RV = 7  # Variance for response time
degradation_mean = -2.379  # Log mean of degradation rate
degradation_stddev = 0.756  # Log standard deviation of degradation rate
e_s_variance = 0.041  # Variance of error term
D_0LLs = 0.352  # Initial DLL_s value after tamping

# Call the sim_seg function
result = sim_seg(
    time_horizon, 
    T_insp, 
    T_tamp, 
    T_step, 
    AL, 
    D_0LLs, 
    degradation_mean, 
    degradation_stddev, 
    e_s_variance, 
    ILL, 
    IALL, 
    RM, 
    RV
)

# Print the result
print(result)