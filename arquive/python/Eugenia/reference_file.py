''' To generate reference file for transformation.

    After getting the reference coordinates in DLC (will be in separate folders), 
    input input/output file paths to generate one file with the reference coordinates'''

from pathlib import Path
import os
import pandas as pd
import glob

# Define the base path
path = Path(r'/Volumes/VS03-NandB-1/Eugenia/FSCV_Rat_Striatum_GoNogo/Data_analysis/R1_4/DLC_pre')
outputpath = Path('/Users/eugeniapoh/Desktop')
# List all folder names in the path
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

subj_dict = {}
for folder in folders:
    ses = os.path.join(path, folder)

    subjectIDs = []

    # List all subjects in the session folder
    subjects = [sub for sub in os.listdir(ses) if os.path.isdir(os.path.join(ses, sub))]

    # Initialize list for session data
    data = []
    ref_coords = []
    
    for sub in subjects:
        file_pattern = os.path.join(ses, sub, '*.csv')
        file_list = glob.glob(file_pattern)

        if len(file_list) != 0:
            # Extract the folder name and subject ID
            folder_name = sub
            subjectID = folder_name.split('_')[0]

            # Process each file found
            for file in file_list:
                # Read the headers
                lines = pd.read_csv(file, nrows=3, header=None)
                # Concatenate from columns 3 onwards
                combined_headers = lines.iloc[1,3:] + '_' + lines.iloc[2,3:]

                df = pd.read_csv(file, skiprows=3, header=None)
                # Insert column headers again
                df.columns = list(lines.iloc[0, :3]) + list(combined_headers)

                # Check if df is non-empty and has the expected columns
            if not df.empty:
                # Add a new column 'SubjectID' as the first column in the DataFrame
                df.insert(0, 'SubjectID', subjectID)
                df.drop(df.columns[[2,3]], axis=1, inplace=True)
                df.rename(columns={'scorer': 'Session'}, inplace=True)

                # Append the processed DataFrame to data list
                data.append(df)
            else:
                print(f"Warning: {file} is empty")
                continue

        else:
            print(f"Warning: No CSV files found for {sub} in session {ses}")
            continue

    # Concatenate all data files for this subject vertically (rows stacked)
    subject_data = pd.concat(data, ignore_index=True, axis=0)

    filename = folder + 'combined_reference_coordinates.csv'
    output_file = os.path.join(outputpath, filename)

    subject_data.to_csv(output_file, index=False)