import pandas as pd
from datetime import datetime, timedelta
import random

def generate_dates(start_date, total_months):
    """Generate an array of dates, one month apart."""
    dates = [start_date + timedelta(days=30 * i) for i in range(total_months)]
    return dates

def generate_dlls_and_tamping(dates, al=1.0, il=1.5, initial_dll=0.5):
    """Generate DLLs, tamping flags, and tamping types according to implantation logic."""
    total_months = len(dates)
    dlls = []
    tamping_performed = []
    tamping_type = []

    current_dll = initial_dll
    for i in range(total_months):
        if i > 0 and i % 15 == 0:  
            current_dll = random.uniform(il + 0.01, il + 0.5)  
        elif i > 0 and i % 6 == 0:  
            current_dll = random.uniform(al, il) 

        dlls.append(current_dll)

        if current_dll > il:  
            tamping_performed.append(1)
            tamping_type.append(2)  
            current_dll = random.uniform(al, il)  
        elif current_dll > al:  
            tamping_performed.append(1)
            tamping_type.append(1) 
            current_dll = random.uniform(0.01, al)  
        else:  # No Tamping
            tamping_performed.append(0)
            tamping_type.append(0)

        if tamping_performed[-1] == 0:
            degradation_rate = random.uniform(0.03, 0.07)
            current_dll += degradation_rate

    return dlls, tamping_performed, tamping_type

def write_to_excel(dates, dlls, tamping_performed, tamping_type, file_path):
    """Write the generated data to an Excel file."""
    data = {
        "Date": [date.strftime('%Y-%m-%d') for date in dates],
        "DLL_s Measurement": dlls,
        "Tamping Performed": tamping_performed,
        "Tamping Type": tamping_type,
    }
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"Data successfully written to {file_path}")

def main():
    start_date = datetime(2002, 1, 1)
    total_months = 100
    output_file = "mock_track_data.xlsx"

    dates = generate_dates(start_date, total_months)

    dlls, tamping_performed, tamping_type = generate_dlls_and_tamping(dates)

    write_to_excel(dates, dlls, tamping_performed, tamping_type, output_file)

if __name__ == "__main__":
    main()