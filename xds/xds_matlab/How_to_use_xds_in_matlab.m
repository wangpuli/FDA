%% First, load the compiled datafile
clc
clear
base_path = 'Z:\limblab\lab_folder\Projects\darpa\Chewie_right\xds\';
file_name = 'Chewie_M1_CO_CS_BL_11032015_002_xds.mat';
load(strcat(base_path, file_name));
% Now you can see there is an 'xds' struct in MATLAB workspace
% An 'xds' structure is generated from Blackrock 'nev' and 'nsx' files. Raw
% spikes has been converted to spike counts with a specific bin width
% (xds.bin_width), while other data (EMG, force, kinematics) are also
% downsampled with the same bin width. Meanwhile, raw spike timings
% can be found in 'xds.spikes'.
%% Here is a basic review of this data structure

% 'xds.meta' contains the basic information of each data file
disp(xds.meta);

% 'xds.bin_width' shows the bin width
disp(xds.bin_width)

% 'xds.time_frame' shows the time frame after binning. 
% These variables: 
% (1) xds.spike_counts
% (2) xds.EMG (if there are EMGs)
% (3) xds.force (if there are forces)
% (4) xds.kin_p (meaning 'kinematics-position')
% (5) xds.kin_v (meaning 'kinematics-velocity')
% (6) xds.kin_a (meaning 'kinematics-acceleration')
% share the same time frame (xds.time_frame)
time_frame = xds.time_frame;

% xds.spike_counts contains the spike counts after binning. Each colum is
% an electrode, and each row is a sample
spike_counts = xds.spike_counts;

% xds.spikes contains the spike timings. Since the number of spikes on each
% electrode is not the same, a MATLAB cell array is used here.
spikes = xds.spikes;

% Sometimes the label of each electrode is important, so xds.unit_names
% stored the label of each electrode (since we don't do spike sorting)
unit_names = xds.unit_names;

% xds.kin_v contains the kinematics-velecity. Each colum is a dimension,
% and each row is a sample
kin_v = xds.kin_v;

% Information about trials are also stored in xds structure. Here just
% three examples. The names of them are quite straightforwrd.
trial_start_time = xds.trial_start_time;
trial_gocue_time = xds.trial_gocue_time;
trial_end_time = xds.trial_end_time;
trial_result = xds.trial_result;

%% Here are some useful functions

% The original .mat file contains binned data. They are usually binned with
% very small bin size, like 0.001 s or 0.01 s. But sometimes you may need 
% larger bin size. To rebin the spike trains, you can use the
% update_bin_data function. Note that this will update all the attributes
% of xds.
% xds = update_bin_data(xds, new_bin_size)

% Re-bin with new_bin_size of 0.05 s
xds = update_bin_data(xds, 0.05);

% if you want to smooth the spike counts, you can use the
% smooth_spike_counts function:
% smoothed_spike_counts = smooth_spike_counts(xds, gaussian kernel sd, sqrt transform or not)

% Using kernel SD = 0.02, no sqrt transform:
smoothed_spike_counts = smooth_spike_counts(xds, 0.02, 'none');

% Using kernel SD = 0.02, do sqrt transform:
smoothed_spike_counts = smooth_spike_counts(xds, 0.02, 'sqrt');

% If you want to extract only the data within successful trials, this
% function can work. Note that you have to specify whether you want to
% segment from "gocue_time" or"start_time" in the second argument of the 
% get_rewarded_trials function.

% The names of the variables it returns are self-explanatory. If there is
% no EMG in this file, for example, the output 'trial_EMG' will be zero.
% Otherwise the data are organized into a cell array, each row of which
% represents a trial. For 'trial_kin', there are 3 colums, the first colum
% is the position, the second colum is the velocity, and the third colum is
% the acceleration.

% In practice, you need matrix to implement algorithms. Notice the outputs of
% this function are all cell arrays, so you may need to convert the cell
% arrays to ordinary matrix. 
start_time = "start_time";  % select either gocue_time or start_time
[trial_spike_counts, trial_EMG, trial_force, trial_kin, trial_tgt_pos] = get_rewarded_trials(xds, start_time);

% You can also extract trial information around force_onset_time like this:
start_event = 'force_onset_time';
end_event = 'force_onset_time';
time_before_start = 0.5;
time_after_end = 1;

if xds.has_force
    time_onset = compute_force_onset_time(xds);
    xds.trial_force_onset_time = time_onset;
end

[trial_spike_counts_onset, trial_emg_onset, trial_force_onset, trial_kin_onset, trial_tgt_pos_onset] = ...
    get_rewarded_trials(xds, start_event, time_before_start, end_event, time_after_end);






