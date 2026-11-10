'''
Eugenia Poh
5 Feb 2025

Processing DLC data for each trialtype

Inputs:
--- fig_dir         --> folder to save figures.
--- path            --> folder with transfromed DLC coordinates as .h5 file (detault output from DLC)
                        Coordinate data should be transformed into cm using matlab's 'fitgeomform2d' function.
--- path_trigger    --> folder with files of trial timestamps for each animal: needs to have a column called 'VideoFrame' as it will cut the 
                        DLC outputs based on this and up to 2 min after the last input. ** important step, since recordings go beyond the actual session
--- output_file_path--> folder to save pkl file. Recommended to make a local file if pulling files from server in case of disconnection.

What is performed here:
1) Exclude trials that were noisy or missing, etc from the final analysis (e.g., if we are averaging the data). Make sure that the order of the DLC and trigger files are correct. 
2) Remove coordinates where likelihood < 0.9
    --> NEW: Interpolate values
3) Remove coordinates where sequential frames > 1.5cm, approx. 45cm/s
4) Calculate speed outliers: 
    - Exclude those in a sliding window of 9 frames where speed values >(mean+2*std)  
    - coordinates during these speed outlier moments were then removed +1 frame at start and end of the moment
    - interpolation using the remaining timepoints were then performed
    --> Interpolate values 
5) X and Y coordinates filtered using a Gaussian-weighted moving average filter (window size = 9)
6) Calculate speed and acceleration for each body part

Outputs:
Pickle file in output_file_path. Accesible by df_name['subjectID']

'''
#%%
import pandas as pd
from pathlib import Path
import numpy as np
import os
import copy
from datetime import datetime
import pickle
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


runMac = 'yes'
pcutoff = 0.9 # pcutoff for the filtering method
filt_method = 3 # 1: just likelihood cut off, 2: big jumps (>1.5cm/frame), 3: sliding window of speed + big jumps
interpolate = 'yes' # yes: interpolate and smoothen, no: no interpolation and smoothing
# session = 'Syed3_noNP'
# session = 'Syed2'
# session = 'GNG_free_delay_noNP'
session = 'GNG_free_delay'

# directories
if runMac == 'yes':
    main_path = Path(r'/Volumes/VS03-NandB-1/Eugenia/FSCV_Rat_Striatum_GoNogo/Data_analysis/R1_4/DLC')
    fig_dir = Path(main_path) / session / 'Figures'
    path = Path(main_path) / session / 'DLC_transformed'
    path_trigger = Path(main_path) / session / 'Trigger_Info_files'

# else: 
    # fig_dir = r'\\vs03\VS03-NandB-3\Willuhn_SBL\SqueakPLaybackVoltammetry\Data_analysis\Behavior Video Analysis\DLC_analysis' # Where to save figures
    # path = Path(r'\\vs03\VS03-NandB-3\Willuhn_SBL\SqueakPLaybackVoltammetry\Data_analysis\Behavior Video Analysis\DLC_transformed')
    # path_trigger = Path(r'\\vs03\VS03-NandB-3\Willuhn_SBL\SqueakPLaybackVoltammetry\Data_analysis\Trigger Info Files')

date_string = datetime.now().strftime("%y%m%d")
filename = f"{session}_{date_string}_filt_df.pkl" # output pickle file name
# output_file_path = Path(fig_dir) / filename
output_file_path = Path(r'/Users/eugeniapoh/Documents/Python/DLC/') / filename # Make this a local directory to save pickle file, incase of server disconnection

#%%
# -----------------------------------------------------------------
#
#               Make dictionaries for DLC coordinates
#
# -----------------------------------------------------------------

# Open transformed DLC coordinates
files = list(path.glob('*.h5')) # transformed DLC coordinates

# Open trigger files
files_trigger = list(path_trigger.glob('*.csv'))

DLCscorer = 'DLC_resnet50_BehSyl projectDec22shuffle1_100000'
# Get the animal IDs
animals = [file.stem for file in files]
animals = [name.replace('_tform_coords', '') for name in animals]

animal_dict = {}
# Iterate through the files to fill the dictionary 
for file in files:
    file_name = file.name  # Get the file name
    animal = file_name.split('_')[0] # Get the ID by extracting everything before the first _
    # Store the animal as the key and the file name as the value in the dictionary
    animal_dict[animal] = file

filenames_trigger = []
for file_trigger in files_trigger:
    filenames_trigger.append(file_trigger.name)
# Combining the DLC and trigger files for each animal to use as a key later
df_paths = pd.DataFrame(list(animal_dict.items()), columns=['Animal', 'path_DLC'])

# Create an empty column 'path_trigger' in path_df
df_paths['path_trigger'] = ''

# Standardize case and strip whitespaces for matching
df_paths['Animal'] = df_paths['Animal']

 # Iterate over each file path in files_trigger
for file in files_trigger:
    animal_id = file.stem.split('_')[0]
    df_paths.loc[df_paths['Animal'] == animal_id, 'path_trigger'] = str(file)

# -----------------------------------------------------------------
#          Check that the files are loaded correctly
# -----------------------------------------------------------------
testdf = pd.read_hdf(files[0])
testdf = testdf[1400:1800]
bodyparts=testdf.columns.get_level_values(1)[::3] #read out the header to get body part names!
print(bodyparts)
bodyparts2plot=bodyparts #you could also take a subset, i.e. =['snout']
scorer = testdf.columns.get_level_values(0)[0]  # Read out the header to get the scorer name!

#%%
# -----------------------------------------------------------------
#
#               Functions
#
# -----------------------------------------------------------------
def get_cmap(n, name='viridis'):
    return plt.get_cmap(name, n)

def readDLCfiles(df_paths, animal='', path_DLC='path_DLC', path_trigger='path_trigger',):
    try:
        # Get the paths
        selected_pathDLC = df_paths.loc[df_paths['Animal'] == animal, path_DLC].values[0]
        selected_pathtrigger = df_paths.loc[df_paths['Animal'] == animal, path_trigger].values[0]

        df_DLC = pd.read_hdf(selected_pathDLC)
        df_trigger = pd.read_csv(selected_pathtrigger)
    except (IndexError, FileNotFoundError, KeyError) as e:
        print(f"Skipping animal {animal} due to missing data: {e}")

    return df_DLC, df_trigger

def getDLCwindow(df_DLC, df_trigger, pre_stimtime=10, post_stimtime=10, StimCategory='', PresentationNumber=1):
    """Get the trajectories for each animal, for each trial type
    
    Parameters:
    - df_DLC: DataFrame containing DLC data.
    - df_trigger: DataFrame containing trigger events.
    - pre_stimtime: Time in seconds to include before stimulus.
    - post_stimtime: Time in seconds to include after stimulus.
    - StimCategory: Category of stimulus to filter (if any).
    - PresentationNumber: The specific presentation number for the stimulus category.
    
    Returns:
    - df_DLC: The modified full DLC DataFrame windowed to the entire trial period.
    - df_subset: A subset of df_DLC centered around the stimulus.
    """
    fps = 29.97
    if not StimCategory:
        # Fine the last trial
        # index = df_trigger.index[-1]
        index = df_trigger['VideoFrame'].idxmax()
        # Determine the frame of the last stimulus
        stimtime = int(df_trigger.loc[index, 'VideoFrame'].item())

        start = int(np.floor(stimtime - (pre_stimtime * fps)))
        stop = int(np.ceil(stimtime + (post_stimtime * fps)))
        
        df_subset = df_DLC[start:stop]
        whole_stop = int(df_trigger.iloc[index]['VideoFrame'].item())

            
    else:
        # Get the index to find the trialtype
        index = df_trigger[(df_trigger['StimCategory'] == StimCategory) & 
                        (df_trigger['PresentationNumber'] == PresentationNumber)].index

        stimtime = int(df_trigger.loc[index, 'VideoFrame'].item())
        start = int(np.floor(stimtime - (pre_stimtime * fps)))
        stop = int(np.ceil(stimtime + (post_stimtime * fps)))

        df_subset = df_DLC[start:stop]
        whole_stop = int(df_trigger.iloc[-1]['VideoFrame'].item())


    # get the last trial and add 120s
    whole_stop = whole_stop + int(120 * fps)
    whole_stop = int(np.ceil(whole_stop))

    df_DLC = df_DLC[:whole_stop]            

    return df_DLC, df_subset  

def getBPspeed(Dataframe, bodyparts2plot):
    ''' 
    Plots the coordinates as linked line and gets the bodypart speeds and acceleration. 
    Returns a DataFrame with speed and acceleration values for specified body parts.
    '''
    scorer = Dataframe.columns.get_level_values(0)[0]  # Read out the header to get the scorer name!
    df_copy = copy.deepcopy(Dataframe)

    bodypart_coords = {}
    # Run this for the raw values
    for bp in bodyparts2plot:
        bodypart_coords[bp] = np.vstack((df_copy[scorer][bp]['x'].values, df_copy[scorer][bp]['y'].values)).T 
        
        # Get the bodypart speed and plot it
        _, bp_speed = getSpeedy(bodypart_coords[bp])
        
        df_copy.loc[:, (scorer, bp, 'speed')] = bp_speed
        df_copy.loc[:, (scorer, bp, 'acceleration')] = np.gradient(bp_speed, 1/29.97)

    return df_copy
    
def getSpeedy(coordinates):   
    # Calculate the differences between consecutive points
    diffs = np.diff(coordinates, axis=0)
    
    # Use numpy's efficient vectorized norm to calculate distances
    distances = np.linalg.norm(diffs, axis=1)

    # Time array based on 29.97 FPS
    time = np.arange(len(distances)+1) / 29.97
    # time = time[1:]

    speed = distances / (1/29.97)
    speed = np.insert(speed, 0,0)

    return time, speed

def smooth_with_gaussian(data, window_size=9):
    ''' 
    Smooths data using a Gaussian window of a given size, similar to MATLAB's smoothdata(A, "gaussian", 9)
    
    Arguments:
    - data: 1D array or pandas Series of data to be smoothed
    - sigma: the size of the Gaussian window, similar to MATLAB's
    '''
    # Generate a Gaussian window
    sigma = (window_size-1) / 6  # To get 99 percentile of gaussian pdf = 6 * sigma
    smoothed_data  = gaussian_filter1d(data, sigma=sigma)
    return smoothed_data 

def getDistance2panel(Dataframe, bodyparts2plot):
    ''' To get the distances to a speific, saving it into the dataframe '''
    scorer = Dataframe.columns.get_level_values(0)[0]  # Read out the header to get the scorer name!
    df_copy = copy.deepcopy(Dataframe)

    # Define p1 and p2 as numpy arrays for vector operations
    p1 = np.array([0, 7])
    p2 = np.array([0, 23.9])
    
    snout = np.array([df_copy[scorer]['snout']['x'], df_copy[scorer]['snout']['y']]).T
    neck = np.array([df_copy[scorer]['neck']['x'], df_copy[scorer]['neck']['y']]).T

    x_cross_range, _, _ = getOrientationAngle(snout, neck) # determine whether the rat is oriented to the magazine

    for bp in bodyparts2plot:
        # Access x and y values from the DataFrame
        x_values = df_copy[scorer][bp]['x']
        y_values = df_copy[scorer][bp]['y']

        # Stack x and y values to form a 2D array of coordinates
        xy = np.stack((x_values, y_values), axis=1)
        
        distances = []
        # Iterate over coordinates and calculate the distance
        for coord in xy:
            coord = np.array(coord)  # Ensure coord is a numpy array for vector operations
            d = (np.cross(p2 - p1, coord - p1) / np.linalg.norm(p2 - p1))
            distances.append(d)
        
        df_copy.loc[:, (scorer, bp, 'distance')] = distances
        df_copy.loc[:, (scorer, bp, 'orientation')] = x_cross_range # add magazine orientation to boolean array

    return df_copy

def getOrientationAngle(xy1, xy2):
    ''' Get the orientation of the rat relative to the speaker. Uses head and neck coordinates
        Key in the coordinates of the panel of interest (x_wall and y_range)  '''   
    x_wall = 0 # x coord of wall with panel of interest

    y_range = [7, 24]
    y_center = [x_wall, (max(y_range)+min(y_range))/2]

    # Calculate orientation angle
    big_angle = np.arctan2(xy2[:, 1] - xy1[:, 1], xy2[:, 0] - xy1[:, 0])  # in radians

    # Convert big_angle to degrees and normalize to 0-360 range
    big_angle_deg = np.degrees(big_angle) % 360  # Convert to degrees and wrap to [0, 360)

    # Calculate the y position where line intersects the x_wall
    x1_wall = x_wall - xy1[:, 0]
    y_crossing = xy1[:, 1] + np.tan(big_angle) * x1_wall  # crossing point along y-axis

    # Check if y_crossing falls within y_range
    x_crossing_in_range = (y_crossing >= y_range[0]) & (y_crossing <= y_range[1])
    condition = xy1[:,0]>xy2[:,0]
    x_crossing_in_range = np.where(condition & x_crossing_in_range, False, x_crossing_in_range)
    
    # Small angle for further use if needed, corrected for y_center
    small_angle = np.arctan2(x1_wall, y_crossing - y_center[1])  # in radians
    small_angle = np.degrees(small_angle) % 360  # Shift by 90 degrees for plotting, convert to degrees and wrap to [0, 360)

    return x_crossing_in_range, small_angle, big_angle_deg

def find_nearest_valid(series, idx, max_search_distance=9):
    """
    Find the nearest valid (non-NaN) value in the series around the given idx.
    Searches to the left and right up to max_search_distance.
    """
    left_idx = idx - 1
    right_idx = idx + 1

    # Search to the left and right within the max_search_distance
    for n in range(1, max_search_distance + 1):
        if left_idx >= 0 and not pd.isna(series.iloc[left_idx]):
            return series.iloc[left_idx]
        if right_idx < len(series) and not pd.isna(series.iloc[right_idx]):
            return series.iloc[right_idx]
        left_idx -= 1
        right_idx += 1
    print(f"Had to use find_nearest_valid")

    # Return NaN if no valid value is found within the search distance
    return float('nan')

def interpolate_nan_blocks(series):
    ''' Find NaN values and interpolate them '''
    series = series.copy()  # Avoid modifying the original data
    nan_indices = np.where(series.isna())[0]  # Find NaN indices
    
    if len(nan_indices) == 0:
        return series  # No NaNs, return as is

    # Find valid indices
    valid_indices = np.where(~series.isna())[0]

    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]

        if end_idx - start_idx > 1:  # There's a gap (NaNs)
            series[start_idx + 1:end_idx] = np.linspace(series[start_idx], series[end_idx], end_idx - start_idx + 1)[1:-1]
    return series

def interpolate_data(keep_indices, x_values, y_values):
    ''' To interpolate the data using linspace '''

    # Interpolate x_values and y_values
    x_values = interpolate_nan_blocks(x_values)
    y_values = interpolate_nan_blocks(y_values)

    # Mark all previously missing indices as valid now
    keep_indices[:] = ~x_values.isna()  # Assuming NaNs are the same in both x and y

    return keep_indices, x_values, y_values

def filter_coord(Dataframe, bodyparts2plot, pcutoff=0.3, method=1, interpolate='yes'):
    f_Dataframe = copy.deepcopy(Dataframe)

    for bp in bodyparts2plot:
        scorer = Dataframe.columns.get_level_values(0)[0]  # Read out the header to get the scorer name!
        x_values = f_Dataframe[scorer][bp]['x'].copy()
        y_values = f_Dataframe[scorer][bp]['y'].copy()
        likelihood = f_Dataframe[scorer][bp]['likelihood'].copy()
        
        keep_indices = np.ones(len(x_values), dtype=bool)
    
        # ------------------------------
        # Filter based on likelihood cutoff
        # ------------------------------
        for idx in range(len(likelihood)):
            if likelihood.iloc[idx] < pcutoff:
                # Mark points with low likelihood for exclusion
                keep_indices[idx] = False

        # Set x and y values to NaN where keep_indices is False
        x_values[~keep_indices] = np.nan
        y_values[~keep_indices] = np.nan

        if interpolate == 'yes':
            keep_indices, x_values_transformed, y_values_transformed = interpolate_data(keep_indices, x_values, y_values)
        else:
            x_values_transformed = x_values
            y_values_transformed = y_values

        # ----------------------------------------------------------------
        # Convert transformed x and y values into arrays for processing
        # ----------------------------------------------------------------
        xy = np.vstack((x_values_transformed, y_values_transformed)).T
        _, bp_speed = getSpeedy(xy)

        if method == 2:
            median_speed = np.nanmedian(bp_speed)
            std_speed = np.nanstd(bp_speed)
            exclude_speed = median_speed + 2 * std_speed

            # ------------------------------
            # Filter based on speed cutoff
            # ------------------------------
            for idx in range(1, len(bp_speed) - 1):
                if bp_speed[idx] > exclude_speed:
                    keep_indices[max(0, idx-1):min(len(bp_speed), idx+2)] = False # Mark t-1, t, and t+1 for exclusion

            # ------------------------------------------------------
            # Filter based on large movements in x and y coordinates
            # ------------------------------------------------------                
            indices_x = np.abs(x_values_transformed.diff()) > 1.5
            indices_y = np.abs(y_values_transformed.diff()) > 1.5
            
            # If the consecutive coordinates are not jumping by more than 2.5cm/frame, keep those
            x_values_transformed[~indices_x] = np.nan
            y_values_transformed[~indices_y] = np.nan

            # merge these indices
            merged_indices = indices_x | indices_y

            # apply the merged indices
            # keep_indices[~merged_indices] = False
            keep_indices &= ~merged_indices  # To accumualte False values

            if interpolate == 'yes':
                # Interpolate a second time        
                keep_indices, x_values_transformed, y_values_transformed = interpolate_data(keep_indices, x_values, y_values)
            
        elif method == 3:
            # ------------------------------------------------------
            # Filter based on large movements in x and y coordinates
            # ------------------------------------------------------                
            indices_x = np.abs(x_values_transformed.diff()) > 1.5
            indices_y = np.abs(y_values_transformed.diff()) > 1.5

            # merge these indices
            merged_indices = indices_x | indices_y

            # apply the merged indices
            # keep_indices[~merged_indices] = False
            keep_indices &= ~merged_indices  # To accumualte False values

            # --------------------------------------------------------------------
            # Filter based on dynamic speed cutoff within the moving window
            # --------------------------------------------------------------------
            window = 9             # Define the window size for the moving filter
            half_window = window // 2

            for idx in range(len(bp_speed)):
                # Define the start and end indices for the window around the current index
                start_idx = max(0, idx - half_window)
                end_idx = min(len(bp_speed), idx + half_window + 1)
                
                # Extract the speed values within the current window
                window_speeds = bp_speed[start_idx:end_idx]
                
                # Calculate local median and standard deviation within the window
                local_mean = np.nanmean(window_speeds)
                local_std = np.nanstd(window_speeds)
                local_cutoff = local_mean + 2 * local_std

            # ---------------------------------------------
            #  Filter based on speed cutoff within window
            # ---------------------------------------------
                # Check if the speed at the current index exceeds the local cutoff
                if bp_speed[idx] > local_cutoff:
                    # Mark the entire window for exclusion if the condition is met
                    keep_indices[start_idx:end_idx] = False     

            if interpolate=='yes':
                # Interpolate a second time        
                keep_indices, x_values_transformed, y_values_transformed = interpolate_data(keep_indices, x_values, y_values)

        # --------------------------------------------------------------------
        # Apply Gaussian smoothing to the transformed x and y coordinates
        # --------------------------------------------------------------------
        # Preserve the index before applying smoothing
        original_index = x_values_transformed.index
        
        smoothed_x_values = smooth_with_gaussian(x_values_transformed, window_size=9)
        smoothed_y_values = smooth_with_gaussian(y_values_transformed, window_size=9)
        
        # Recreate pandas Series with the original index to preserve row titles
        smoothed_x_values = pd.Series(smoothed_x_values, index=original_index)
        smoothed_y_values = pd.Series(smoothed_y_values, index=original_index)
        
        # --------------------------------------------------------------------
        # Update the f_Dataframe with the filtered and smoothed values
        # --------------------------------------------------------------------
        # Use .loc to update the x and y values in the DataFrame
        f_Dataframe.loc[:, (scorer, bp, 'x')] = smoothed_x_values
        f_Dataframe.loc[:, (scorer, bp, 'y')] = smoothed_y_values
        f_Dataframe.loc[:, (scorer, bp, 'time')] = range(1, len(f_Dataframe) + 1)
        
    # Return the shape of the interpolated data
    return f_Dataframe

# -----------------------------------------------------------------
#
#               To sort out the file paths
#
# -----------------------------------------------------------------
# Create data frame with directories for running scripts
df_DLC = {}
df_trigger = {}

selected_triggers = 'path_trigger'

for index, row in df_paths.iterrows():
    path_trigger = row[selected_triggers]
    animal = animals[index]
    
    if not path_trigger:  # Check if 'path_trigger' is empty
        print(f"Skipping for {animal} as there is no analysed video")
        continue
    else:
        # Grab all the file information
        df_DLC[animal], df_trigger[animal] = readDLCfiles(df_paths, animal=animal)
#%%
all_filtered_Dataframe = {}

for animal in animals:
    try:
    # ------------------------------------------------
    #        Plot the filtered by whole rec data
    # ------------------------------------------------   
        print(f"Filtering {animal}")
        df_filtered = filter_coord(df_DLC[animal], bodyparts2plot, pcutoff=pcutoff, method=filt_method, interpolate=interpolate) 
        df_filtered, _ = getDLCwindow(df_filtered, df_trigger[animal])
        all_filtered_Dataframe[animal] = getBPspeed(df_filtered, bodyparts2plot) # get the speed and acceleration values
        all_filtered_Dataframe[animal] = getDistance2panel(all_filtered_Dataframe[animal], bodyparts2plot) # Get the distance to the panel and whether its oriented to the magazine

    except Exception:
        print(f"Could not pickle {animal}")
        continue
    # ------------------------------------------------
    #        Pickle the data
    # ------------------------------------------------   
    with open(output_file_path, 'wb') as file:
        pickle.dump(all_filtered_Dataframe, file)
    
#%%
