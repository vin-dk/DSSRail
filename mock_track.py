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
    

def track_create(num_tracks, start_date=datetime(2002, 1, 1), total_months=100):
    """
    Generate mock track data for multiple sections of track and store them in a dedicated directory.
    
    Parameters:
        num_tracks (int): Number of track sections (Excel files) to generate.
        start_date (datetime): Start date for the simulation.
        total_months (int): Duration in months for each track section.
    """
    home_dir = os.path.expanduser("~")
    track_data_dir = os.path.join(home_dir, "track_data")

    if not os.path.exists(track_data_dir):
        os.makedirs(track_data_dir)
        print(f"Directory created: {track_data_dir}")
    else:
        print(f"Directory already exists: {track_data_dir}")

    for i in range(1, num_tracks + 1):
        print(f"Generating data for Track Section {i}...")
        dates = generate_dates(start_date, total_months)
        dlls, tamping_performed, tamping_type = generate_dlls_and_tamping(dates)

        file_name = f"mock_track_data_{i}.xlsx"
        file_path = os.path.join(track_data_dir, file_name)

        write_to_excel(dates, dlls, tamping_performed, tamping_type, file_path)

    print(f"\nAll {num_tracks} track sections generated successfully in: {track_data_dir}")

def main():
    start_date = datetime(2002, 1, 1)
    total_months = 100
    output_file = "mock_track_data.xlsx"

    dates = generate_dates(start_date, total_months)

    dlls, tamping_performed, tamping_type = generate_dlls_and_tamping(dates)

    write_to_excel(dates, dlls, tamping_performed, tamping_type, output_file)
    


if __name__ == "__main__":
    main()