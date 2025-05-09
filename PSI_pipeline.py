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
    # NOTE, inverted from DLL_s
    import pandas as pd
    import numpy as np
    from scipy.stats import anderson

    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string pointing to an Excel file.")

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")

    if df.empty:
        raise ValueError("Input file is empty.")

    required_columns = ["Date", "PSI", "Maintenance", "Maintenance Action"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")

    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        raise ValueError("Failed to parse 'Date' column as datetime.")

    df = df.sort_values(by="Date")

    maintenance_indices = df[df["Maintenance"] == 1].index.tolist()
    if len(maintenance_indices) < 2:
        raise ValueError("At least two maintenance events are required to calculate degradation.")

    degradation_rates = []
    time_intervals = []

    for i in range(len(maintenance_indices) - 1):
        start_idx = maintenance_indices[i] + 1
        end_idx = maintenance_indices[i + 1]

        if end_idx <= start_idx or end_idx >= len(df):
            continue

        sub_df = df.loc[start_idx:end_idx]
        if sub_df.empty:
            continue

        psi_start = sub_df["PSI"].iloc[0]
        psi_end = sub_df["PSI"].iloc[-1]
        days = (sub_df["Date"].iloc[-1] - sub_df["Date"].iloc[0]).days

        if days <= 0:
            continue

        rate = (psi_start - psi_end) / days  # PSI drops over time → degradation is positive
        degradation_rates.append(rate)
        time_intervals.append(days)

    psi_diffs = df["PSI"].diff().dropna()
    psi_diffs = -psi_diffs[psi_diffs < 0]  # Invert to keep only actual PSI drops (positive degradation)

    if psi_diffs.empty:
        raise ValueError("Not enough valid positive degradation differences to test for lognormality.")

    log_psi_values = np.log(psi_diffs)
    result = anderson(log_psi_values, dist="norm")

    if result.statistic < result.critical_values[2]:
        print("PSI degradation rates follow a lognormal distribution.")
        log_mean = np.mean(log_psi_values)
        log_std = np.std(log_psi_values)

        degradation_mean = np.exp(log_mean + 0.5 * log_std**2)
        degradation_stddev = np.sqrt((np.exp(log_std**2) - 1) * np.exp(2 * log_mean + log_std**2))
    else:
        print("PSI degradation rates do not follow a lognormal distribution. Assuming normal distribution.")
        degradation_mean = psi_diffs.mean()
        degradation_stddev = psi_diffs.std()

    print(f"Degradation Mean: {degradation_mean}")
    print(f"Degradation Stddev: {degradation_stddev}")

    most_recent_maintenance = df[df["Maintenance"] == 1].iloc[-1]
    most_recent_index = most_recent_maintenance.name

    if most_recent_index + 1 < len(df):
        PSI_0 = df.iloc[most_recent_index + 1]["PSI"]
    else:
        PSI_0 = most_recent_maintenance["PSI"]

    print(f"PSI_0 (Initial PSI after last maintenance): {PSI_0}")

    return {
        "PSI_0": PSI_0,
        "degradation_mean": degradation_mean,
        "degradation_stddev": degradation_stddev,
        "degradation_rates": degradation_rates,
        "time_intervals": time_intervals,
    }


def degrade_PSI(current_PSI, b_s, time, e_s):
    next_PSI = current_PSI - (b_s * time) - e_s  # PSI decreases over time
    return max(next_PSI, 0)  # Ensures PSI does not drop below 0


def recovery_PSI(current_PSI, maintenance_action):
    """
    Simple, rule-based PSI recovery function.

    Args:
        current_PSI (float): PSI value before recovery
        maintenance_action (int): Tamping type
            1 = Minor
            2 = Moderate
            3 = Major
            4 = Full reset

    Returns:
        float: Recovered PSI (capped at 1.0)
    """
    if maintenance_action == 4:
        return 1.0  # Full reset, per Guler et al.

    # Define recovery ratios (can tweak at runtime)
    recovery_ratios = {
        1: 0.10,  # Minor: recover 10% of lost PSI
        2: 0.30,  # Moderate: recover 30% of lost PSI
        3: 0.50,  # Major: recover 50% of lost PSI
    }

    if maintenance_action not in recovery_ratios:
        raise ValueError(f"Invalid Maintenance Action: {maintenance_action}")

    lost = 1.0 - current_PSI
    recovered = recovery_ratios[maintenance_action] * lost
    return min(current_PSI + recovered, 1.0)


def sim_seg_PSI(
    time_horizon, T_insp, T_tamp, T_step, AL, PSI_0,
    lognormal_mean, lognormal_stddev,
    e_s_mean, e_s_stddev,
    recovery_thresholds=(0.6, 0.4, 0.2)
):
    t = 1
    tn = 0
    Npm = Ncm_n = Ncm_e = Ninsp = 0
    PSI_t = PSI_0

    b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)

    while t <= time_horizon:
        t += T_step
        e_s = np.random.normal(e_s_mean, e_s_stddev)
        PSI_t = max(PSI_t - (b_s * T_step) - e_s, 0)

        if t % T_tamp == 0 and PSI_t <= AL:
            PSI_t = recovery_PSI(PSI_t, maintenance_action=1)
            Npm += 1
            tn = t
            b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)

        if t % T_insp == 0:
            Ninsp += 1
            if recovery_thresholds[0] < PSI_t <= 1.0:
                continue
            elif recovery_thresholds[1] < PSI_t <= recovery_thresholds[0]:
                PSI_t = recovery_PSI(PSI_t, maintenance_action=2)
                Ncm_n += 1
                tn = t
                b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)
            elif recovery_thresholds[2] < PSI_t <= recovery_thresholds[1]:
                PSI_t = recovery_PSI(PSI_t, maintenance_action=3)
                Ncm_n += 1
                tn = t
                b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)
            elif 0.0 <= PSI_t <= recovery_thresholds[2]:
                PSI_t = recovery_PSI(PSI_t, maintenance_action=4)
                Ncm_e += 1
                tn = t
                b_s = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_stddev)

    return {
        "final_t": t,
        "final_PSI": PSI_t,
        "b_s": b_s,
        "Npm": Npm,
        "Ncm_n": Ncm_n,
        "Ncm_e": Ncm_e,
        "Number of inspections": Ninsp
    }


def monte_PSI(
    time_horizon, T_insp, T_tamp, T_step, AL, PSI_0,
    degradation_mean, degradation_stddev,
    e_s_mean, e_s_stddev,
    num_simulations,
    inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost,
    recovery_thresholds=(0.6, 0.4, 0.2)
):
    total_inspections = total_pm = total_cm_n = total_cm_e = total_cost = 0

    for _ in range(num_simulations):
        result = sim_seg_PSI(
            time_horizon, T_insp, T_tamp, T_step, AL, PSI_0,
            degradation_mean, degradation_stddev,
            e_s_mean, e_s_stddev,
            recovery_thresholds
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

    return {
        "Total Inspections": total_inspections,
        "Total Preventive Maintenances": total_pm,
        "Total Normal Corrective Maintenances": total_cm_n,
        "Total Emergency Corrective Maintenances": total_cm_e,
        "Total Cost": total_cost,
        "Average Inspections": total_inspections / num_simulations,
        "Average Preventive Maintenances": total_pm / num_simulations,
        "Average Normal Corrective Maintenances": total_cm_n / num_simulations,
        "Average Emergency Corrective Maintenances": total_cm_e / num_simulations,
        "Average Cost": total_cost / num_simulations
    }

def run_multi_segment_monte(
    PSI_0_list,
    time_horizon, T_insp, T_tamp, T_step, AL,
    degradation_mean, degradation_stddev,
    e_s_mean, e_s_stddev,
    num_simulations,
    inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost
):
    """
    Runs monte_PSI for a list of PSI_0 starting values with shared parameters.

    Returns a list of result dictionaries.
    """
    results = []

    for idx, PSI_0 in enumerate(PSI_0_list):
        print(f"[INFO] Running simulation for segment {idx + 1} with PSI_0 = {PSI_0:.4f}")
        result = monte_PSI(
            time_horizon, T_insp, T_tamp, T_step, AL, PSI_0,
            degradation_mean, degradation_stddev,
            e_s_mean, e_s_stddev,  # These map to re_s_m and re_s_s but are used for e_s
            RM=0, RV=0,  # Placeholder since PSI model doesn't currently use response time
            num_simulations=num_simulations,
            inspection_cost=inspection_cost,
            preventive_maintenance_cost=preventive_maintenance_cost,
            normal_corrective_maintenance_cost=normal_corrective_maintenance_cost,
            emergency_corrective_maintenance_cost=emergency_corrective_maintenance_cost
        )
        result["Segment Index"] = idx + 1
        result["Starting PSI"] = PSI_0
        results.append(result)

    return results

import pandas as pd


def export_multi_segment_monte_results_to_excel(
    PSI_0_list,
    time_horizon, T_insp, T_tamp, T_step, AL,
    degradation_mean, degradation_stddev,
    e_s_mean, e_s_stddev,
    num_simulations,
    inspection_cost, preventive_maintenance_cost,
    normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost,
    output_file="monte_psi_results.xlsx"
):
    """
    Runs multi-segment Monte Carlo simulations and exports the results to an Excel file.
    """
    results = run_multi_segment_monte(
        PSI_0_list,
        time_horizon, T_insp, T_tamp, T_step, AL,
        degradation_mean, degradation_stddev,
        e_s_mean, e_s_stddev,
        num_simulations,
        inspection_cost, preventive_maintenance_cost,
        normal_corrective_maintenance_cost, emergency_corrective_maintenance_cost
    )

    df = pd.DataFrame(results)
    df = df[
        [
            "Segment Index", "Starting PSI",
            "Average Inspections", "Average Preventive Maintenances",
            "Average Normal Corrective Maintenances", "Average Emergency Corrective Maintenances",
            "Average Cost"
        ]
    ]

    df.to_excel(output_file, index=False)
    print(f"[INFO] Monte Carlo results exported to: {output_file}")
    return df

def full_pipeline_from_load_to_monte(
    input_load_file,
    traffic_load_per_year,
    material_type,
    output_dir=".",
    num_simulations=1000,
    time_horizon=365,
    T_insp=30,
    T_tamp=60,
    T_step=1,
    AL=0.7,
    e_s_mean=0.0,
    e_s_stddev=0.01,
    inspection_cost=100,
    preventive_maintenance_cost=300,
    normal_corrective_maintenance_cost=700,
    emergency_corrective_maintenance_cost=1500,
    recovery_thresholds=(0.6, 0.4, 0.2)
):
    import os

    # Step 1: Convert Load File → PSI File
    psi_df = process_load_file_to_psi(
        input_file=input_load_file,
        traffic_load_per_year=traffic_load_per_year,
        material_type=material_type
    )

    # Step 2: Derive initial PSI and degradation stats
    stats = take_in_PSI(input_load_file.replace("Load", "converted"))
    PSI_0 = stats["PSI_0"]
    degradation_mean = stats["degradation_mean"]
    degradation_stddev = stats["degradation_stddev"]

    # Step 3: Run Monte Carlo for this one segment
    results = run_multi_segment_monte(
        psi_list=[PSI_0],
        num_simulations=num_simulations,
        time_horizon=time_horizon,
        T_insp=T_insp,
        T_tamp=T_tamp,
        T_step=T_step,
        AL=AL,
        degradation_mean=degradation_mean,
        degradation_stddev=degradation_stddev,
        e_s_mean=e_s_mean,
        e_s_stddev=e_s_stddev,
        inspection_cost=inspection_cost,
        preventive_maintenance_cost=preventive_maintenance_cost,
        normal_corrective_maintenance_cost=normal_corrective_maintenance_cost,
        emergency_corrective_maintenance_cost=emergency_corrective_maintenance_cost,
        recovery_thresholds=recovery_thresholds
    )

    # Step 4: Save results to Excel
    base_name = os.path.splitext(os.path.basename(input_load_file))[0]
    out_path = os.path.join(output_dir, f"{base_name}_monte_results.xlsx")
    export_multi_segment_monte_results_to_excel(results, out_path)

    print(f"Full pipeline complete. Results saved to: {out_path}")
    return results

def full_pipeline_multi_load_to_monte(
    load_file_list,
    traffic_load_per_year,
    material_type,
    output_dir=".",
    num_simulations=1000,
    time_horizon=365,
    T_insp=30,
    T_tamp=60,
    T_step=1,
    AL=0.7,
    e_s_mean=0.0,
    e_s_stddev=0.01,
    inspection_cost=100,
    preventive_maintenance_cost=300,
    normal_corrective_maintenance_cost=700,
    emergency_corrective_maintenance_cost=1500,
    recovery_thresholds=(0.6, 0.4, 0.2)
):
    all_results = {}

    for file_path in load_file_list:
        print(f"\n[•] Processing file: {file_path}")
        try:
            result = full_pipeline_from_load_to_monte(
                input_load_file=file_path,
                traffic_load_per_year=traffic_load_per_year,
                material_type=material_type,
                output_dir=output_dir,
                num_simulations=num_simulations,
                time_horizon=time_horizon,
                T_insp=T_insp,
                T_tamp=T_tamp,
                T_step=T_step,
                AL=AL,
                e_s_mean=e_s_mean,
                e_s_stddev=e_s_stddev,
                inspection_cost=inspection_cost,
                preventive_maintenance_cost=preventive_maintenance_cost,
                normal_corrective_maintenance_cost=normal_corrective_maintenance_cost,
                emergency_corrective_maintenance_cost=emergency_corrective_maintenance_cost,
                recovery_thresholds=recovery_thresholds
            )
            all_results[os.path.basename(file_path)] = result

        except Exception as e:
            print(f"[!] Failed to process {file_path}: {e}")

    print("\n[✓] All load files processed.")
    return all_results

def full_pipeline_from_psi_to_monte(
    input_psi_file,
    output_dir=".",
    num_simulations=1000,
    time_horizon=365,
    T_insp=30,
    T_tamp=60,
    T_step=1,
    AL=0.7,
    e_s_mean=0.0,
    e_s_stddev=0.01,
    inspection_cost=100,
    preventive_maintenance_cost=300,
    normal_corrective_maintenance_cost=700,
    emergency_corrective_maintenance_cost=1500,
    recovery_thresholds=(0.6, 0.4, 0.2)
):
    from pathlib import Path

    # 1. Derive degradation and PSI_0
    degradation_data = take_in_PSI(input_psi_file)

    PSI_0 = degradation_data["PSI_0"]
    degradation_mean = degradation_data["degradation_mean"]
    degradation_stddev = degradation_data["degradation_stddev"]

    # 2. Run Monte Carlo
    results = monte_PSI(
        time_horizon=time_horizon,
        T_insp=T_insp,
        T_tamp=T_tamp,
        T_step=T_step,
        AL=AL,
        PSI_0=PSI_0,
        degradation_mean=degradation_mean,
        degradation_stddev=degradation_stddev,
        e_s_mean=e_s_mean,
        e_s_variance=e_s_stddev,
        re_s_m=e_s_mean,
        re_s_s=e_s_stddev,
        RM=0,
        RV=0,
        num_simulations=num_simulations,
        inspection_cost=inspection_cost,
        preventive_maintenance_cost=preventive_maintenance_cost,
        normal_corrective_maintenance_cost=normal_corrective_maintenance_cost,
        emergency_corrective_maintenance_cost=emergency_corrective_maintenance_cost
    )

    # 3. Export results
    base_name = Path(input_psi_file).stem
    output_path = os.path.join(output_dir, f"{base_name}_monte_results_from_PSI.xlsx")
    pd.DataFrame([results]).to_excel(output_path, index=False)

    print(f"[✓] Monte results from PSI saved to: {output_path}")
    return results


import os
import pandas as pd
from pathlib import Path

def batch_pipeline_from_psi_files_to_monte(
    input_psi_files,
    output_path="batch_psi_to_monte_results.xlsx",
    num_simulations=1000,
    time_horizon=365,
    T_insp=30,
    T_tamp=60,
    T_step=1,
    AL=0.7,
    e_s_mean=0.0,
    e_s_stddev=0.01,
    inspection_cost=100,
    preventive_maintenance_cost=300,
    normal_corrective_maintenance_cost=700,
    emergency_corrective_maintenance_cost=1500,
    recovery_thresholds=(0.6, 0.4, 0.2)
):
    all_results = []

    for psi_file in input_psi_files:
        try:
            print(f"[•] Processing PSI file: {psi_file}")
            result = full_pipeline_from_psi_to_monte(
                input_psi_file=psi_file,
                output_dir=".",  # temp individual result if needed
                num_simulations=num_simulations,
                time_horizon=time_horizon,
                T_insp=T_insp,
                T_tamp=T_tamp,
                T_step=T_step,
                AL=AL,
                e_s_mean=e_s_mean,
                e_s_stddev=e_s_stddev,
                inspection_cost=inspection_cost,
                preventive_maintenance_cost=preventive_maintenance_cost,
                normal_corrective_maintenance_cost=normal_corrective_maintenance_cost,
                emergency_corrective_maintenance_cost=emergency_corrective_maintenance_cost,
                recovery_thresholds=recovery_thresholds
            )
            result["Segment"] = Path(psi_file).stem
            all_results.append(result)
        except Exception as e:
            print(f"[!] Failed to process {psi_file}: {e}")

    df = pd.DataFrame(all_results)
    df.to_excel(output_path, index=False)
    print(f"[✓] Batch Monte results from PSI saved to: {output_path}")
    return df


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


def process_load_file_to_psi(input_file, traffic_load_per_year, material_type, output_file="PSI_converted.xlsx"):
    import pandas as pd
    import numpy as np

    # Material constants: age limit in years × traffic (MGT/year) = ML
    AGE_LIMITS = {1: 45, 2: 40, 3: 25, 4: 30, 5: 18, 6: 21}
    KF = 5.2

    if material_type not in AGE_LIMITS:
        raise ValueError("Invalid material type. Must be one of: 1, 2, 3, 4, 5, 6")

    ML = AGE_LIMITS[material_type] * traffic_load_per_year

    def calculate_psi(cumulative_load, kf=KF, ml=ML):
        return 1 - np.exp(kf * ((cumulative_load / ml) - 1))

    def inverse_psi(psi, kf=KF, ml=ML):
        if psi >= 1.0:
            return 1.0
        return ml * (1 + (1 / kf) * np.log(1 - psi))

    # Load and verify structure
    df = pd.read_excel(input_file)
    expected_cols = ["Date", "Cumulative Load", "Maintenance", "Maintenance Action"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Input file must contain columns: {expected_cols}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    # Recovery effect ratios
    recovery_ratios = {
        1: 0.10,
        2: 0.30,
        3: 0.50,
        4: 1.00  # Full reset
    }

    psi_values = []
    memory_load = df.iloc[0]["Cumulative Load"]

    for i, row in df.iterrows():
        raw_load = row["Cumulative Load"]
        maintenance = row["Maintenance"]
        action = row["Maintenance Action"]

        # Step 1: compute PSI based on the current "memory" load
        psi = calculate_psi(memory_load)

        # Step 2: if maintenance is performed, adjust PSI
        if maintenance == 1:
            if action not in recovery_ratios:
                raise ValueError(f"Invalid Maintenance Action at index {i}: {action}")

            recover_frac = recovery_ratios[action]
            if recover_frac == 1.0:
                psi = 1.0
                memory_load = 1.0  # Reset to baseline
            else:
                lost = 1.0 - psi
                psi = min(psi + recover_frac * lost, 1.0)
                memory_load = inverse_psi(psi)

        # Save PSI
        psi_values.append(psi)

        # Update memory load using delta between current and previous raw load
        if i + 1 < len(df):
            next_raw = df.iloc[i + 1]["Cumulative Load"]
            delta = next_raw - raw_load
            memory_load += delta

    # Add PSI to DataFrame
    df["PSI"] = psi_values
    output_df = df[["Date", "PSI", "Maintenance", "Maintenance Action"]]
    output_df.to_excel(output_file, index=False)

    print(f"Processed PSI file saved as: {output_file}")
    return output_df

import os

def batch_process_load_files_to_psi(
    folder_path, traffic_load_per_year, material_type, output_folder=None
):
    """
    Processes all Excel files in a folder, converting load data to PSI.

    Args:
        folder_path (str): Path to the folder containing input Excel files.
        traffic_load_per_year (float): Annual traffic load in MGT/year.
        material_type (int): Material type (1 to 6).
        output_folder (str, optional): Folder to save output files. Defaults to folder_path.
    """
    if output_folder is None:
        output_folder = folder_path

    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path is not a valid folder: {folder_path}")

    input_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".xls", ".xlsx")) and not f.startswith("~$")
    ]

    if not input_files:
        raise ValueError("No Excel files found in the specified folder.")

    for file in input_files:
        input_path = os.path.join(folder_path, file)
        output_name = f"PSI_{file}"
        output_path = os.path.join(output_folder, output_name)

        try:
            print(f"Processing {file}...")
            process_load_file_to_psi(
                input_file=input_path,
                traffic_load_per_year=traffic_load_per_year,
                material_type=material_type,
                output_file=output_path
            )
        except Exception as e:
            print(f"[!] Failed to process {file}: {e}")

    print("Batch processing complete.")