''' To transform DLC outputs for further processing

    Transform the DLC data based on the tform matrix file (generated in MATLAB)
    and reference coordinates '''

from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
import pandas as pd

# session = 'Syed3_noNP'
# session = 'Syed2'
session = 'GNG_free_delay_noNP'

# path to raw DLC output folders with .h5, labelled videos, .csv files, etc. 
path = Path(r'/Volumes/VS03-NandB-1/Eugenia/FSCV_Rat_Striatum_GoNogo/Data_analysis/R1_4/DLC')
# path = Path(r'\\vs03\VS03-NandB-3\Willuhn_SBL\SqueakPLaybackVoltammetry\Data_analysis\Behavior Video Analysis\DLC')
# path = Path(r'C:\Users\poh\OneDrive - Netherlands Institute for Neuroscience\Desktop\TMP_DLC\Updating DLC') # for running a few to update because of errors in video
DLC_raw_outputs = Path(path) / session / 'DLC_raw_outputs'
DLC_files = list(DLC_raw_outputs.glob('*.h5'))

# t_form matrix path folder
ref_points_path = Path(path) / session / 'tform_matrices'
tform_files = list(ref_points_path.glob('*.csv'))

# define export locations
# exportloc = r'\\vs03\VS03-NandB-3\Willuhn_SBL\SqueakPLaybackVoltammetry\Data_analysis\Behavior Video Analysis\DLC_transformed'
# exportloc = r'C:\Users\poh\OneDrive - Netherlands Institute for Neuroscience\Desktop\TMP_DLC\Updating DLC\tform_coords'
exportpath = Path(path) / session / 'DLC_transformed'

# ------------------------------------------------------------------------------------------------------------------------------


DLCscorer = 'DLC_resnet50_BehSyl projectDec22shuffle1_100000'

filenames = []
for file in DLC_files:
    filenames.append(file.name)  # Append the file name to the list
    # print(filenames)


# define the transformations
def get_cmap(n, name='hsv'):
    return plt.get_cmap(name, n)

def transformpoints(Dataframe, bodyparts2plot, transformation_matrix):
    ''' Transform the xy coordinates based on a transformation matrix.'''
    scorer = Dataframe.columns.get_level_values(0)[0]  # Read out the header to get the scorer name!
    
    # Initialize a new DataFrame with the same structure as the original DataFrame
    df_transformed = copy.deepcopy(Dataframe)

    for bpindex, bp in enumerate(bodyparts2plot):
        # extract X and Y coordinates for bodypart
        x_coords = Dataframe[scorer][bp]['x'].values
        y_coords = Dataframe[scorer][bp]['y'].values

        # prepare coordinates in homogenous form
        homogeneous_coords = np.vstack([x_coords, y_coords, np.ones(len(x_coords))])
        # Apply the transformation matrix
        transformed_coords = transformation_matrix @ homogeneous_coords

        # Extract the transformed X and Y (divide by the third row for perspective transformation)
        x_transformed = transformed_coords[0, :] / transformed_coords[2, :]
        y_transformed = transformed_coords[1, :] / transformed_coords[2, :]

        # Assign the transformed coordinates back to the copied DataFrame using .loc
        df_transformed.loc[:, (scorer, bp, 'x')] = x_transformed
        df_transformed.loc[:, (scorer, bp, 'y')] = y_transformed

    return df_transformed

# Get the name of the DLC_files
for ratID_list in range(len(DLC_files)):
    # Get the last part of the path
    ratID = filenames[ratID_list].split('\\')[-1] 
    ratID = ratID.split('_')[0]
    print(ratID)

    # Read the dataframe
    df = pd.read_hdf(DLC_files[ratID_list])

    # Dictionary to map ratID to CSV file
    tform_files_dict = {file.stem: file for file in tform_files}

    # Check if the CSV corresponding to the ratID exists
    if ratID in tform_files_dict:
        tform_csv_path = tform_files_dict[ratID]
        tform_df = pd.read_csv(tform_csv_path, header=None)
        tform = tform_df.to_numpy()
    else:
        print(f"No CSV file found for ratID: {ratID}")

    df_subset = df
    bodyparts=df_subset.columns.get_level_values(1) #read out the header to get body part names! gets all bodyparts
    bodyparts2plot=bodyparts #you could also take a subset, i.e. =['snout']
    # bodyparts2plot=['snout']

    df_transformed = transformpoints(df_subset, bodyparts2plot, tform)

    # Combine the export path and filename
    exportname = exportpath / f'{ratID}_tform_coords.h5'
    df_transformed.to_hdf(exportname, key='df', mode='w')
    exportname = exportpath /  f'{ratID}_tform_coords.csv'
    df_transformed.to_csv(exportname)

    