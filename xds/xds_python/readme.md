# Python codes for loading xds datasets

## Overview
This repository contains Python codes to load neural activity data stored in `.mat` file as xds structure. 

After parsing the data file, a `lab_data` object will be constructed in memory. Then one can grab different types of data asscociated with this data file and do their own preprocessing (re-binning or down-sampling) by calling the attributes or functions of the `lab_data` object.

## Setup
#### Environment:
Python 3

#### Dependencies:
numpy, scipy, pickle

#### How to install
After cloning this repo to your local machine, add the following lines at the beginning of your script:
```
import fnmatch, os, sys
sys.path.append('your path to /xds/xds_python')
```

## A typical work flow
Following the codes below in your own programs
```
import numpy as np
import fnmatch, os
from xds import lab_data
import pickle

bin_size = 0.05 # The unit is second
smooth_window_size = 0.1 # The unit is second

# Specifying the data path, and find out the .mat file under the path
# The program can hadle different formats of the .mat file (v7.3 or not) automatically
base_path = 'your_path/'
mat_list = np.sort(fnmatch.filter(os.listdir(base_path), "*.mat"))

# Load the data file
my_lab_data = lab_data(base_path, mat_list[0])
print(my_lab_data.get_meta())

# Get spike counts for each trial and save as a list
spike_counts = my_lab_data.get_trials_data_spike_counts('R', 'start_time', 0, 'end_time', 0)

# Bin the spikes to get firing rates
my_lab_data.update_bin_data(bin_size)
 
# Do the smoothing with a Gaussian kernel 
my_lab_data.smooth_binned_spikes(bin_size, 'gaussian', smooth_window_size)

# Altenatively, you can do the smoothing with a causal-Gaussian kernel
my_lab_data.smooth_binned_spikes(bin_size, 'half_gaussian', smooth_window_size)

# Get EMGs for each trial and save as a list
EMG = my_lab_data.get_trials_data_EMG('R', 'start_time', 0, 'end_time', 0)

# Get cursor related kinematics for each trial and save as 3 lists
cursor_position, cursor_velocity, cursor_acc = my_lab_data.get_trials_data_cursor('R', 'start_time', 0, 'end_time', 0)

# Get forces for each trial and save as a list
force = my_lab_data.get_trials_data_force('R', 'gocue_time', 0, 'end_time', 0)
```




## Run the codes
#### Loading the data file and construct `lab_data` object
An example `.mat` data file (`Chewie_20150713_001.mat`) is already under the `/data` directory. 

To load this data file, run:
```
base_path = './'
file_name = 'Chewie_20131022_001.mat'
dataset = lab_data(base_path, file_name)
```
Now we have a `lab_data` instance named **`'dataset'`** in the memory. In fact you can name the instance whatever you like.
#### Checking the basic information of the data file
The basic information of the data file is stored in the attribute of `__meta`. You cannot visit it directly because it is designed to be a **private** attribute. To get access to the dataset information stored in `__meta`, call this function:
```
dataset_info = dataset.get_meta()
```
It will return a `dictionary array` containining basic information of this data file.

You can also print the information directly by calling:
```
dataset.print_file_info()
```
#### Grabbing raw spike timings
Raw spike timings are stored as an attribute called `spikes` in `lab_data` object. You can visit this attribute directly by running:
```
spikes = dataset.spikes
```
It will return a `list`. Each row of the list is a `numpy array` containing the raw spike timings, corresponds to a unit (sorted) or an unsorted channel. 

To get the names of each unit / channel, run this:
```
unit_names = dataset.unit_names
```

#### Grabbing binned spike counts
For binned data it's important to know the information about the time course and the binning information. So before grabbing the actual data, run this first:
```
time_frame = dataset.time_frame
bin_width = dataset.bin_width
```
Here`'time_frame'` is a `numpy array` containing the time frame for all binned / downsampled data. So it is also the time frame for EMGs (if any), forces (if any) and kinematics (if any).

To get the spike counts, run this:
```
spike_counts = dataset.spike_counts
unit_names = dataset.unit_names
```
Here `'spike_counts'` is a 2D `numpy` array with shape `(number of samples, number of channels)`, and the time frame for the samples can be obtained by running `time_frame = dataset.time_frame`

#### Grabbing EMG
Since not every dataset has EMG, you should check it first. To make things easier, there's an attribute which can be visited directly called `'has_EMG'`. If `'has_EMG'` is 1, it means there are some EMG data in this file, while if `'has_EMG'` is 0, there's no EMG in this file. Run this to grab EMG data:
```
if dataset.has_EMG == 1:
    EMG = dataset.EMG
    EMG_names = dataset.EMG_names
else:
    print('This file does not contrain EMG')  
```
Here `'EMG'` is a 2D `numpy` array with shape `(number of samples, number of channels)`. As mentioned above, the time frame for EMG is also `time_frame = dataset.time_frame`.

#### Grabbing forces:
Similar to EMG, there's another attribute called `'has_force'`. Run this to get the force data:
```
if dataset.has_force == 1:
    force = dataset.force
```
Again the time frame for force data is `time_frame = dataset.time_frame`.

#### Grabbing kinematics
The attribute `'has_kin'` indicates whether there's any kinematics data in this file. The attribute `'kin_p'` stores position data, and `'kin_v'` stores velocity data, while `'kin_a'` contains acceleration data. Run this to grab all kinematics data:
```
if dataset.has_kin == 1:
    kin_p = dataset.kin_p
    kin_v = dataset.kin_v
    kin_a = dataset.kin_a
```
#### Get trial related data
Up to now, we are still working without trials. But there are also a bunch of functions which can help to extract data asscociated with each trial.

* To extract **spike counts** within specific trials, run this:
    ```
    trial_spike_counts = dataset.get_trials_data_spike_counts('R')
    ``` 
* To extract **EMG** within specific trials, run this:
    ```
    trial_EMG = dataset.get_trials_data_EMG('R')
    ``` 
* To extract **force** within specific trials, run this:
    ```
    trial_force = dataset.get_trials_data_force('R')
    ``` 
* To extract **kinematics** within specific trials, run this:
    ```
    trial_kin_p, trial_kin_v, trial_kin_a = dataset.get_trials_data_kin('R')
    ``` 
* Further, to extract **time frame** for specific trials, run this:
    ```
    trial_time_frame = dataset.get_trials_data_time_frame('R')
    ``` 
The returning of these functions are all `lists`, each row of which corresponds to a trial. Here `'R'` means rewarded trials. You can also use `'A'` and `'F'` to extract `aborted` or `failed` trials if you are interested in them. In fact `'R'` has been set as the default for all these functions.

Sometimes we need different time points to define the start of a trial, like the actual start time (usually it's the **Center On** time), or the go cue time. The trial functions above provide a way to do this:
```
trial_spike_counts = dataset.get_trials_data_spike_counts('R', 'gocue_time', 0.5)
trial_EMG = dataset.get_trials_data_EMG('R', 'gocue_time', 0.5)
trial_kin_p, trial_kin_v, trial_kin_a = dataset.get_trials_data_kin('R', 'gocue_time', 0.5)
```
These codes above means we are defining 0.5 s ahead from the go cue time as our trial start moment.

The default settings for those functions are still using the actual trial start time. So
```
trial_spike_counts = dataset.get_trials_data_spike_counts('R', 'start_time', 0)
```
and
```
trial_spike_counts = dataset.get_trials_data_spike_counts('R')
```
still mean that we extract trials from the actual trial start time.

#### More for trial related information
Here are some other useful trial related attributes and functions.

For target directed lab tasks, information like the position and the direction of the targets, trial start time, go cue time and trial end time are most important. So these information are stored as separate attributes of the `'lab_data'` object, which makes them easier to visit. You can run these codes to grab these information:
```
target_corners = dataset.trial_target_corners
target_dir = dataset.trial_target_dir
trial_result = dataset.trial_result
start_time = dataset.trial_start_time
end_time = dataset.trial_end_time
gocue_time = dataset.trial_gocue_time
```
Meanwhile, to get such trial related information for a specific type of trials, you can run:
```
summary = dataset.get_trials_summary('R')
```
Here `'summary'` is a dictionary array, in which the trial information listed above will be stored corresponding to each trial.

The trial related information listed above are important, but besides them there are also some other trial related information. Here some attributes and functions are provided to check them. By running:
```
dataset.print_trial_info_table_header()
```
You can print the header for the raw trial information table. Then you can check the printed contents and decide whether you need any extra information. Let's say you find there is an entry named `'bump_time'` in the header and you need it for your analysis, then you can grab `'bump_time'` data by running:
```
bump_time = get_one_colum_in_trial_info_table('bump_time')
```
Here the returning of this function is a `list`.

#### Re-binning the data and save extra pickle files 
The original `.mat` file contains binned data. They are usually binned with very small bin size, like `0.001 s` or  `0.01 s`. But sometimes you may need larger bin size. This can be supported by a series of re-binning or downsampling function.

* Re-binning the spike trains
    Run this to re-bin the spike trains:
    ```
    time_spike_counts, spike_counts = dataset.bin_spikes(0.02)
    ```
    Here `'time_spike_counts'` is the new time frame for the re-binned spike train, and `'spike_counts'` is a `'numpy'` array.

* Downsampling EMG data:
    ```
    if dataset.has_EMG == 1:
        time_EMG, EMG = dataset.resample_EMG(0.02)
    ```
    Here `'time_EMG'` is the new time frame for resampled `EMG`, and `EMG` is a `numpy` array.

* Downsampling force data:
    ```
    time_force, force = dataset.resample_force(0.02)
    ```
    Here `'time_force'` is the new time frame for resampled force.

* Downsampling kinematics data:
    ```
    time_kin, kin_p, kin_v, kin_a = dataset.resample_kin(0.02)
    ```
    Here `'time_kin'` is the new time frame for the kinematics.

You may notice that after binning each type of data has its own time_frame. By running the functions above alone will only return new time frames and re-binned data, but the time frame and the attributes in `lab_data` object are still unchanged. These functions above provide you methods to do the re-binnnig separately. But here **an extra function is also provided to do the re-binning and down-sampling together**. If you decide to re-bin the data and update the `lab_data` instance, **you don't need to run the functions listed above, you only need to run this**:
```
dataset.update_bin_data(0.02)
print(dataset.bin_width)
```
The name of this function is self-explanatory. It doesn't have any returning. Instead, attributes including `'time_frame'`, `'spike_counts'`, `'EMG'`, `'force'`, `'kin_p'`, `'kin_v'`, `'kin_a'` will all be substituted by the re-binned one. In this case, all variables share the same new `'time_frame'`.    

#### Save to pickle file
By running:
```
dataset.save_to_pickle('./new_path/', 'new_file_name')
```
You can save current `lab_data` instance to a pickle file.

If you want to load the pickle file, run:
```
import _pickle as pickle
with open ('path_and_file_name', 'rb') as fp:
    my_data = pickle.load(fp)  
spike_counts = my_data.spike_counts 
```




