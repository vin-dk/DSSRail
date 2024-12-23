import numpy as np

# Constants from the case study
alpha_1 = -1.2596
beta_1 = 0.8875
beta_2 = -1.2034
beta_3 = 0.1334

degradation_rate_mean = 0.07
degradation_rate_stddev = 0.06

C_0 = 6.926
C_1 = 0.0208
b = -4.0457

# Initial DLL_s value
DLL_s = 0.99

# Helper funcs
def degrade(current_DLL_s, degradation_rate_mean, degradation_rate_stddev):
    b_s = np.random.lognormal(mean=np.log(degradation_rate_mean), sigma=degradation_rate_stddev)
    epsilon_s = np.random.normal(0, degradation_rate_stddev)
    return max(0, current_DLL_s + b_s + epsilon_s)

def recover(DLL_s, tamping_type):
    epsilon_LL = np.random.normal(0, np.sqrt(0.15))
    R_LL_s = (
        alpha_1 +
        beta_1 * DLL_s +
        beta_2 * tamping_type +
        beta_3 * tamping_type * DLL_s +
        epsilon_LL
    )
    return (DLL_s - R_LL_s)

def defect_probability(DLL_s):
    P_leq_1 = np.exp(C_0 + b * DLL_s) / (1 + np.exp(C_0 + b * DLL_s))
    P_leq_2 = np.exp(C_1 + b * DLL_s) / (1 + np.exp(C_1 + b * DLL_s))
    P1 = P_leq_1
    P2 = P_leq_2 - P_leq_1
    P3 = 1 - P_leq_2
    return {"P1": P1, "P2": P2, "P3": P3}

# Sim params
iterations = 80000
DLL_s_values = []
defect_probabilities = []

# Constants for limits
AL = 1.5 
IL = 2.0 
IAL = 3.0  

# counters
preventive_count = 0
partial_count = 0
complete_count = 0

# sim loop
for i in range(iterations):
    DLL_s = degrade(DLL_s, degradation_rate_mean, degradation_rate_stddev)

    if DLL_s > AL and DLL_s <= IL:
        # Preventive maintenance (complete tamping)
        DLL_s = recover(DLL_s, tamping_type=1)
        preventive_count += 1
    elif DLL_s > IL and DLL_s <= IAL:
        # Normal corrective maintenance (partial tamping)
        DLL_s = recover(DLL_s, tamping_type=0)
        partial_count += 1
    elif DLL_s > IAL:
        # Emergency corrective maintenance (complete tamping)
        DLL_s = recover(DLL_s, tamping_type=1)
        complete_count += 1

    # Calculate defect probabilities
    probs = defect_probability(DLL_s)

    # Record results
    DLL_s_values.append(DLL_s)
    defect_probabilities.append(probs)

    # Print iteration details
    print(f"Iteration {i + 1}: DLL_s = {DLL_s}, P1 = {probs['P1']}, P2 = {probs['P2']}, P3 = {probs['P3']}")
    

# Aggregate 
avg_P1 = np.mean([p["P1"] for p in defect_probabilities])
avg_P2 = np.mean([p["P2"] for p in defect_probabilities])
avg_P3 = np.mean([p["P3"] for p in defect_probabilities])
avg_DLL_s = np.mean(DLL_s_values)
lowest_DLL_s = np.min(DLL_s_values)
highest_DLL_s = np.max(DLL_s_values)
    
print("\nAggregate Results:")
print(f"Average P1: {avg_P1}")
print(f"Average P2: {avg_P2}")
print(f"Average P3: {avg_P3}")
print(f"Average DLL_s: {avg_DLL_s}")
print(f"Lowest DLL_s: {lowest_DLL_s}")
print(f"Highest DLL_s: {highest_DLL_s}")
print(f"Preventive Maintenance Count: {preventive_count}")
print(f"Partial Maintenance Count: {partial_count}")
print(f"Complete Maintenance Count: {complete_count}")
print(f"Total Maintenance Actions: {preventive_count + partial_count + complete_count}")


