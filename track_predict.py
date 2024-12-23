import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import os
import matplotlib.pyplot as plt

def take_in(file_path):
    data = pd.read_excel(file_path)
    
    required_columns = ["Date", "DLL_s Measurement", "Tamping Performed", "Tamping Type"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")
    
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by="Date")
    results = {}

    ## 1. Compute D0LL_s
    most_recent_tamping = data[data["Tamping Performed"] == 1]
    if most_recent_tamping.empty:
        raise ValueError("No tamping data available to compute initial degradation.")
    
    most_recent_tamping_index = most_recent_tamping.index[-1]
    
    # Check if there is a row after the most recent tamping
    if most_recent_tamping_index + 1 < len(data):
        D0LL_s = data.loc[most_recent_tamping_index + 1, "DLL_s Measurement"]
    else:
        # If no row is available after the most recent tamping, use the tamping row itself
        D0LL_s = data.loc[most_recent_tamping_index, "DLL_s Measurement"]
    
    results["D0LL_s"] = D0LL_s

    ## 2. Degradation Rates
    tamping_indices = data[data["Tamping Performed"] == 1].index.tolist()
    if len(tamping_indices) < 2:
        raise ValueError("Insufficient tamping data to calculate degradation rates.")
    
    degradation_rates = []
    for i in range(len(tamping_indices) - 1):
        start_row = data.loc[tamping_indices[i] + 1] if (tamping_indices[i] + 1 < len(data)) else None
        end_row = data.loc[tamping_indices[i + 1]]
        
        if start_row is not None:
            delta_DLL = end_row["DLL_s Measurement"] - start_row["DLL_s Measurement"]
            delta_time = (end_row["Date"] - start_row["Date"]).days
            if delta_time > 0:  
                degradation_rate = delta_DLL / delta_time
                degradation_rates.append(degradation_rate)
    
    if not degradation_rates:
        raise ValueError("No valid degradation rates found.")
    
    results["degradation_rate_mean"] = np.mean(degradation_rates)
    results["degradation_rate_stddev"] = np.std(degradation_rates)

    ## 3. Defect Probability Coefficients
    AL, IL, IAL = 1.5, 2.0, 3.0 #  real lim: 1.5, 2.0, 3.0
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


def degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev):
    """
    Models the degradation progression over a given time interval.
    """
    b_s = np.random.lognormal(mean=np.log(degradation_rate_mean), sigma=degradation_rate_stddev)
    
    epsilon_s = np.random.normal(0, degradation_rate_stddev)  
    
    next_DLL_s = current_DLL_s + b_s + epsilon_s
    
    return max(0, next_DLL_s)  

def defect(current_DLL_s, defect_coefficients):
    """
    Calculates defect probabilities using ordinal logistic regression.
    """
    C_0 = defect_coefficients["C_0"]
    C_1 = defect_coefficients["C_1"]
    b = defect_coefficients["b"]
    
    # Calculate cumulative probabilities
    P_leq_1 = np.exp(C_0 + b * current_DLL_s) / (1 + np.exp(C_0 + b * current_DLL_s))
    P_leq_2 = np.exp(C_1 + b * current_DLL_s) / (1 + np.exp(C_1 + b * current_DLL_s))
    
    # Convert to individual probabilities
    P1 = P_leq_1
    P2 = P_leq_2 - P_leq_1
    P3 = 1 - P_leq_2
    
    return {"P1": P1, "P2": P2, "P3": P3}


def recovery_alt(current_DLL_s, recovery_coefficients, tamping_type):
    """
    Alternative recovery logic that mirrors the behavior described in the paper.
    If corrective maintenance (CM) is critical, the recovery has a stronger effect.
    """
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]

    noise_stddev = np.sqrt(0.15)
    epsilon_LL = np.random.normal(0, noise_stddev)

    R_LL_s = (
        alpha_1 +
        beta_1 * current_DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * current_DLL_s +
        epsilon_LL
    )

    updated_DLL_s = max(0.1, current_DLL_s - R_LL_s)
    return updated_DLL_s

def compute_alt(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t, 
                defect_coefficients, recovery_coefficients, schedule_months, 
                current_month, AL=1.0, IL=1.5):
    """
    Alternative compute method implementing PM at regular intervals and CM only for critical defect probabilities.
    """
    AL = 1.5
    IL = 2.0
    IAL = 3.0
    # Step 1: Degrade
    next_DLL_s_before_recovery = degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev)
    degradation_val = next_DLL_s_before_recovery - current_DLL_s

    # Step 2: Defect probabilities
    defect_probs = defect(next_DLL_s_before_recovery, defect_coefficients)

    # Determine if PM or CM is needed
    tamping_status = None
    recovery_adjustment = 0
    updated_DLL_s = next_DLL_s_before_recovery
    failure = defect_probs['P2'] > 0.75
    critical_failure = defect_probs['P3'] > 0.05  # High probability of critical defect

    # Scheduled Preventive Maintenance (PM)
    if current_month % schedule_months == 0 and next_DLL_s_before_recovery > AL:
        tamping_status = "complete"  # PM happens at the scheduled interval
        updated_DLL_s = recovery_alt(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=1)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s
        
    
     # Emergency CM
    if next_DLL_s_before_recovery > IAL or critical_failure:
        tamping_status = "complete (emergency)"
        updated_DLL_s = recovery_alt(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=1)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s   

    # Corrective Maintenance (CM)
    elif next_DLL_s_before_recovery > IL or failure:
        tamping_status = "partial"  # CM for critical failures
        updated_DLL_s = recovery_alt(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=0)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s
        

    # Return results for the simulation step
    return {
        "current_DLL_s": current_DLL_s,
        "degradation_val": degradation_val,
        "recovery_adjustment": recovery_adjustment,
        "updated_DLL_s": updated_DLL_s,
        "defect_probabilities": defect_probs,
        "tamping_status": tamping_status
    }

def sim_alt(file_path, interval_months, time_length_months, schedule_months):
    """
    Simulates the degradation and recovery process with scheduled preventive maintenance
    and critical corrective maintenance, including cost tracking.

    Parameters:
    - file_path: str, path to the input Excel file.
    - interval_months: int, the time interval in months between each step.
    - time_length_months: int, total simulation time length in months.
    - schedule_months: int, regular tamping schedule interval in months.

    Returns:
    - mass_data: list, containing data for all runs and intervals.
    """
    # Cost parameters (in USD, converted from SEK based on the paper)
    inspection_cost = 24  # per interval
    preventive_maintenance_cost = 453  # PM (scheduled tamping)
    corrective_maintenance_partial_cost = 1000  # CM (partial)
    corrective_maintenance_complete_cost = 3600  # CM (emergency complete)

    total_intervals = time_length_months // interval_months
    delta_t_days = interval_months * 30

    mass_data = []
    input_data = take_in(file_path)
    current_DLL_s = input_data["D0LL_s"]
    degradation_rate_mean = input_data["degradation_rate_mean"]
    degradation_rate_stddev = input_data["degradation_rate_stddev"]
    recovery_coefficients = input_data["recovery_coefficients"]
    defect_coefficients = input_data["defect_coefficients"]

    # Cost tracking variables
    cumulative_cost = 0
    inspection_total_cost = 0
    pm_total_cost = 0
    cm_partial_total_cost = 0
    cm_emergency_total_cost = 0

    # Tamping counts
    tamping_counts = {
        "none": 0,
        "partial": 0,
        "complete": 0,
        "complete_emergency": 0
    }

    degradation_values = []
    recovery_values = []
    probabilities = {"P1": [], "P2": [], "P3": []}

    for interval in range(total_intervals):
        current_month = interval * interval_months
        print(f"Run {interval + 1}")

        # Inspection cost for this interval
        interval_inspection_cost = inspection_cost
        inspection_total_cost += interval_inspection_cost

        # Use the alternative compute logic
        step_results = compute_alt(
            current_DLL_s,
            degradation_rate_mean,
            degradation_rate_stddev,
            delta_t_days,
            defect_coefficients,
            recovery_coefficients,
            schedule_months,
            current_month
        )

        interval_data = {
            "run": interval + 1,
            "current_DLL_s": step_results["current_DLL_s"],
            "degradation_val": step_results["degradation_val"],
            "recovery_adjustment": step_results["recovery_adjustment"],
            "updated_DLL_s": step_results["updated_DLL_s"],
            "defect_probabilities": step_results["defect_probabilities"],
            "tamping_status": step_results["tamping_status"],
        }

        # Add tamping costs based on the tamping status
        interval_pm_cost = 0
        interval_cm_partial_cost = 0
        interval_cm_emergency_cost = 0

        if step_results["tamping_status"] == "complete":
            interval_pm_cost = preventive_maintenance_cost
            pm_total_cost += interval_pm_cost
            tamping_counts["complete"] += 1
        elif step_results["tamping_status"] == "partial":
            interval_cm_partial_cost = corrective_maintenance_partial_cost
            cm_partial_total_cost += interval_cm_partial_cost
            tamping_counts["partial"] += 1
        elif step_results["tamping_status"] == "complete (emergency)":
            interval_cm_emergency_cost = corrective_maintenance_complete_cost
            cm_emergency_total_cost += interval_cm_emergency_cost
            tamping_counts["complete_emergency"] += 1

        interval_total_cost = (
            interval_inspection_cost
            + interval_pm_cost
            + interval_cm_partial_cost
            + interval_cm_emergency_cost
        )
        cumulative_cost += interval_total_cost

        interval_data.update({
            "inspection_cost": interval_inspection_cost,
            "pm_cost": interval_pm_cost,
            "cm_partial_cost": interval_cm_partial_cost,
            "cm_emergency_cost": interval_cm_emergency_cost,
            "total_cost": interval_total_cost,
            "cumulative_cost": cumulative_cost,
        })

        mass_data.append(interval_data)
        current_DLL_s = step_results["updated_DLL_s"]

        # Update aggregated metrics
        degradation_values.append(step_results["degradation_val"])
        recovery_values.append(step_results["recovery_adjustment"])
        probabilities["P1"].append(step_results["defect_probabilities"]["P1"])
        probabilities["P2"].append(step_results["defect_probabilities"]["P2"])
        probabilities["P3"].append(step_results["defect_probabilities"]["P3"])

        if step_results["tamping_status"] is None:
            tamping_counts["none"] += 1

        for key, value in interval_data.items():
            print(f"{key}: {value}")
        print("\n")

    # Compute averages
    avg_degradation = np.mean(degradation_values)
    avg_recovery = np.mean(recovery_values)
    avg_probabilities = {key: np.mean(values) for key, values in probabilities.items()}
    
    # Convert recovery_coefficients to normal numbers
    recovery_coefficients_cleaned = {key: float(value) for key, value in recovery_coefficients.items()}
    
    # Convert defect_coefficients to normal numbers
    defect_coefficients_cleaned = {key: float(value) for key, value in defect_coefficients.items()}
    
    # Print session data
    print("\nSession Data:")
    print(f"degradation_rate_mean: {round(degradation_rate_mean, 2)}")
    print(f"degradation_rate_stddev: {round(degradation_rate_stddev, 2)}")
    print("recovery_coefficients:", recovery_coefficients_cleaned)
    print("defect_coefficients:", defect_coefficients_cleaned)    
    print(f"Tamping counts: {tamping_counts}")
    print(f"Average degradation: {round(avg_degradation, 2)}")
    print(f"Average recovery value: {round(avg_recovery, 2)}")
    print(f"Average P1: {round(avg_probabilities['P1'], 2)}")
    print(f"Average P2: {round(avg_probabilities['P2'], 2)}")
    print(f"Average P3: {round(avg_probabilities['P3'], 2)}")

    # Print cost summary
    print("\nSession Cost Data:")
    print(f"Total inspection cost: ${inspection_total_cost:.2f}")
    print(f"Total preventive maintenance cost: ${pm_total_cost:.2f}")
    print(f"Total partial CM cost: ${cm_partial_total_cost:.2f}")
    print(f"Total emergency CM cost: ${cm_emergency_total_cost:.2f}")
    print(f"Cumulative total cost: ${cumulative_cost:.2f}")

    return mass_data

def analyze_cost_vs_maintenance_limits(file_path, maintenance_limits, interval_months, time_length_months, schedule_months):
    """
    Analyze the effect of maintenance limits on total costs.

    Parameters:
        file_path (str): Path to the input data file.
        maintenance_limits (list): List of maintenance limits to evaluate (AL, IL, IAL).
        interval_months (int): Time interval in months for each step in the simulation.
        time_length_months (int): Total simulation length in months.
        schedule_months (int): Regular interval for scheduled maintenance in months.

    Returns:
        None
    """
    results = []

    for AL, IL, IAL in maintenance_limits:
        print(f"Analyzing maintenance limits AL={AL}, IL={IL}, IAL={IAL}...")
        data = sim_alt(file_path, interval_months, time_length_months, schedule_months)
        total_cost = sum(run["total_cost"] for run in data)
        results.append((AL, IL, IAL, total_cost))

    
    for result in results:
        print(f"AL={result[0]}, IL={result[1]}, IAL={result[2]} -> Total Cost: ${result[3]:,.2f}")

    
    maintenance_limits_str = [f"AL={AL}, IL={IL}, IAL={IAL}" for AL, IL, IAL in maintenance_limits]
    total_costs = [result[3] for result in results]

    plt.figure(figsize=(10, 6))
    plt.bar(maintenance_limits_str, total_costs, color="skyblue")
    plt.xlabel("Maintenance Limits")
    plt.ylabel("Total Maintenance Cost (USD)")
    plt.title("Cost vs. Maintenance Limits")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    

def analyze_maintenance_action_frequencies(file_path, maintenance_limits, interval_months, time_length_months, schedule_months):
    """
    Analyze the frequency of maintenance actions (preventive, corrective, emergency) under different maintenance limits.

    Parameters:
        file_path (str): Path to the input data file.
        maintenance_limits (list): List of maintenance limits to evaluate (AL, IL, IAL).
        interval_months (int): Time interval in months for each step in the simulation.
        time_length_months (int): Total simulation length in months.
        schedule_months (int): Regular interval for scheduled maintenance in months.

    Returns:
        None
    """
    results = []

    for AL, IL, IAL in maintenance_limits:
        print(f"Analyzing maintenance limits AL={AL}, IL={IL}, IAL={IAL}...")
        data = sim_alt(file_path, interval_months, time_length_months, schedule_months)
        preventive = sum(run["tamping_status"] == "complete" for run in data)
        corrective = sum(run["tamping_status"] == "partial" for run in data)
        emergency = sum(run["tamping_status"] == "complete (emergency)" for run in data)
        results.append((AL, IL, IAL, preventive, corrective, emergency))

    
    for result in results:
        print(f"AL={result[0]}, IL={result[1]}, IAL={result[2]} -> "
              f"Preventive: {result[3]}, Corrective: {result[4]}, Emergency: {result[5]}")

    
    maintenance_limits_str = [f"AL={AL}, IL={IL}, IAL={IAL}" for AL, IL, IAL in maintenance_limits]
    preventive_actions = [result[3] for result in results]
    corrective_actions = [result[4] for result in results]
    emergency_actions = [result[5] for result in results]

    x = np.arange(len(maintenance_limits_str))  
    width = 0.2  

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, preventive_actions, width, label="Preventive", color="green")
    plt.bar(x, corrective_actions, width, label="Corrective", color="orange")
    plt.bar(x + width, emergency_actions, width, label="Emergency", color="red")
    plt.xlabel("Maintenance Limits")
    plt.ylabel("Number of Maintenance Actions")
    plt.title("Maintenance Action Frequencies")
    plt.xticks(x, maintenance_limits_str, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

file_path = r"C:\Users\13046\mock_track_data.xlsx"  


maintenance_limits = [
    # test cases for maintenance limit methods 
    (1.5, 2.0, 3.0),  # 1: Standard limits
    (1.2, 1.8, 2.8),  # 2: Tighter limits
    (1.7, 2.2, 3.2),  # 3: Looser limits
]


analyze_cost_vs_maintenance_limits(file_path, maintenance_limits, interval_months=4, time_length_months=180, schedule_months=12)


analyze_maintenance_action_frequencies(file_path, maintenance_limits, interval_months=4, time_length_months=180, schedule_months=12)

sim_alt(file_path, 4, 1000, 6)

# The below methods simulate ONLY that recovery happens, as necessary, with no regards to a particular tamping schedule. This is a simplistic outlook, and only applies tamping as necessary, as 
# limits are exceeded.



# The above version are alternate methods that consider a given timeschedule of regularly scheduled maintenence. In this case, we define when regularly scheduled preventative maintenence is performed
# and attempt to perform corrections according to this schedule, UNLESS critical defect probability is high. 


def recovery(current_DLL_s, recovery_coefficients, tamping_type):
    """
    Models the recovery of degradation after a tamping event.
    """
    alpha_1 = recovery_coefficients["alpha_1"]
    beta_1 = recovery_coefficients["beta_1"]
    beta_2 = recovery_coefficients["beta_2"]
    beta_3 = recovery_coefficients["beta_3"]
    
    noise_stddev=np.sqrt(0.15)
    
    epsilon_LL = np.random.normal(0, noise_stddev)
    
    R_LL_s = (
        alpha_1 +
        beta_1 * current_DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * current_DLL_s +
        epsilon_LL
    )
    
    updated_DLL_s = max(0, current_DLL_s - R_LL_s)  
    
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
    next_DLL_s_before_recovery = degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev, delta_t)
    degradation_val = next_DLL_s_before_recovery - current_DLL_s  # Calculate degradation value

    defect_probs = defect(next_DLL_s_before_recovery, defect_coefficients)

    tamping_status = None
    recovery_adjustment = 0  
    updated_DLL_s = next_DLL_s_before_recovery 
    if next_DLL_s_before_recovery > IL:
        tamping_status = "partial"
        updated_DLL_s = recovery(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=0)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s
    elif AL < next_DLL_s_before_recovery <= IL:
        # Complete tamping (PM, preventive)
        tamping_status = "complete"
        updated_DLL_s = recovery(next_DLL_s_before_recovery, recovery_coefficients, tamping_type=1)
        recovery_adjustment = next_DLL_s_before_recovery - updated_DLL_s

    return {
        "current_DLL_s": current_DLL_s,
        "degradation_val": degradation_val,
        "recovery_adjustment": recovery_adjustment, 
        "updated_DLL_s": updated_DLL_s,
        "defect_probabilities": defect_probs,
        "tamping_status": tamping_status
    }

def sim(file_path, interval_months, time_length_months):
    """
    Simulates the degradation and recovery process for the specified time length
    with a given interval, including cost tracking.

    Parameters:
    - file_path: str, path to the input Excel file.
    - interval_months: int, the time interval in months between each step.
    - time_length_months: int, the total simulation time length in months.

    Returns:
    - mass_data: list, containing data for all runs and intervals.
    """
    # Cost parameters (in USD, converted from SEK based on the paper)
    inspection_cost = 24  # per interval
    preventive_maintenance_cost = 720  # PM (scheduled tamping)
    corrective_maintenance_complete_cost = 1080  # CM (emergency complete tamping)
    corrective_maintenance_partial_cost = 540  # CM (emergency partial tamping)

    total_intervals = time_length_months // interval_months
    delta_t_days = interval_months * 30 

    mass_data = []

    input_data = take_in(file_path)
    current_DLL_s = input_data["D0LL_s"]
    degradation_rate_mean = input_data["degradation_rate_mean"]
    degradation_rate_stddev = input_data["degradation_rate_stddev"]
    recovery_coefficients = input_data["recovery_coefficients"]
    defect_coefficients = input_data["defect_coefficients"]

    recovery_coefficients_cleaned = {key: round(float(value), 2) for key, value in recovery_coefficients.items()}

    tamping_counts = {"none": 0, "partial": 0, "complete": 0}
    degradation_values = []
    recovery_values = []
    probabilities = {"P1": [], "P2": [], "P3": []}

    # Cost tracking variables
    cumulative_cost = 0
    inspection_total_cost = 0
    pm_total_cost = 0
    cm_total_cost = 0

    for interval in range(total_intervals):
        print(f"Run {interval + 1}")

        # Inspection cost for this interval
        interval_inspection_cost = inspection_cost
        inspection_total_cost += interval_inspection_cost

        step_results = compute(
            current_DLL_s,
            degradation_rate_mean,
            degradation_rate_stddev,
            delta_t_days,
            defect_coefficients,
            recovery_coefficients
        )

        interval_data = {
            "run": interval + 1,
            "current_DLL_s": step_results["current_DLL_s"],
            "degradation_val": step_results["degradation_val"],
            "recovery_adjustment": step_results["recovery_adjustment"],  # Show recovery adjustment
            "updated_DLL_s": step_results["updated_DLL_s"],
            "defect_probabilities": step_results["defect_probabilities"],
            "tamping_status": step_results["tamping_status"],
        }

        # Add tamping costs based on the tamping status
        interval_pm_cost = 0
        interval_cm_cost = 0
        if step_results["tamping_status"] == "complete":
            interval_pm_cost = preventive_maintenance_cost
            pm_total_cost += interval_pm_cost
        elif step_results["tamping_status"] == "partial":
            interval_cm_cost = corrective_maintenance_partial_cost
            cm_total_cost += interval_cm_cost
        elif step_results["tamping_status"] == "emergency_complete":
            interval_cm_cost = corrective_maintenance_complete_cost
            cm_total_cost += interval_cm_cost

        interval_total_cost = interval_inspection_cost + interval_pm_cost + interval_cm_cost
        cumulative_cost += interval_total_cost

        interval_data.update({
            "inspection_cost": interval_inspection_cost,
            "pm_cost": interval_pm_cost,
            "cm_cost": interval_cm_cost,
            "total_cost": interval_total_cost,
            "cumulative_cost": cumulative_cost,
        })

        mass_data.append(interval_data)
        current_DLL_s = step_results["updated_DLL_s"]

        degradation_values.append(step_results["degradation_val"])
        recovery_values.append(step_results["recovery_adjustment"])
        probabilities["P1"].append(step_results["defect_probabilities"]["P1"])
        probabilities["P2"].append(step_results["defect_probabilities"]["P2"])
        probabilities["P3"].append(step_results["defect_probabilities"]["P3"])
        tamping_status = step_results["tamping_status"]
        if tamping_status is None:
            tamping_counts["none"] += 1
        else:
            tamping_counts[tamping_status] += 1

        for key, value in interval_data.items():
            print(f"{key}: {value}")
        print("\n") 

    avg_degradation = sum(degradation_values) / len(degradation_values) if degradation_values else 0
    avg_recovery = sum(recovery_values) / len(recovery_values) if recovery_values else 0
    avg_probabilities = {key: sum(values) / len(values) if values else 0 for key, values in probabilities.items()}

    defect_coefficients_cleaned = {key: round(float(value), 2) for key, value in defect_coefficients.items()}

    print("\nSession Data:")
    print(f"degradation_rate_mean: {round(degradation_rate_mean, 2)}")
    print(f"degradation_rate_stddev: {round(degradation_rate_stddev, 2)}")
    print(f"recovery_coefficients: {recovery_coefficients_cleaned}")
    print(f"defect_coefficients: {defect_coefficients_cleaned}")
    print(f"tamping_counts: {tamping_counts}")
    print(f"Average degradation: {round(avg_degradation, 2)}")
    print(f"Average recovery value: {round(avg_recovery, 2)}")
    print(f"Average P1: {round(avg_probabilities['P1'], 2)}")
    print(f"Average P2: {round(avg_probabilities['P2'], 2)}")
    print(f"Average P3: {round(avg_probabilities['P3'], 2)}")

    # Print cost summary
    print("\nSession Cost Data:")
    print(f"Total inspection cost: ${inspection_total_cost:.2f}")
    print(f"Total preventive maintenance cost: ${pm_total_cost:.2f}")
    print(f"Total corrective maintenance cost: ${cm_total_cost:.2f}")
    print(f"Cumulative total cost: ${cumulative_cost:.2f}")

    return mass_data

def sim_track(directory_path, interval_months, time_length_months):
    """
    Simulates the degradation and recovery process for multiple tracks, represented by Excel files
    in a given directory.

    Parameters:
    - directory_path: str, path to the directory containing Excel files for multiple tracks.
    - interval_months: int, the time interval in months between each step.
    - time_length_months: int, the total simulation time length in months.

    Returns:
    - results: dict, containing simulation results for each track segment.
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"The directory '{directory_path}' does not exist.")
    
    excel_files = [file for file in os.listdir(directory_path) if file.endswith(('.xlsx', '.xls'))]
    if not excel_files:
        raise ValueError(f"No Excel files found in the directory '{directory_path}'.")

    results = {}

    for i, file_name in enumerate(excel_files, start=1):
        file_path = os.path.join(directory_path, file_name)
        print(f"Track Segment {i}: {file_name}") 
        try:
            segment_results = sim(file_path, interval_months, time_length_months)
            results[f"Track Segment {i}"] = segment_results
        except Exception as e:
            print(f"Error processing '{file_name}': {e}")
            continue

    return results