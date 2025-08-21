function xds = xds_from_pklcage(pklpath)

%this function takes a .mat file that was created from a pkl cage file using scipy.io.savemat and
%turns it into an XDS struct. 

load(pklpath)

spike_counts = [];
EMG = [];
time_frame = [];
raw_EMG = [];
raw_EMG_time_frame = [];
FSR_data = [];
behavior = {};
behavior_start_time = [];
behavior_end_time = [];
EMG_names = {};
unit_names = {};
raw_EMG_fs = file{1}.raw_EMG_fs;
neurons = size(file{1}.spike_timing, 2);
spikes = cell(1, neurons);

for i = 1:length(file{1}.EMG_names)
    EMG_names{1, i} = file{1}.EMG_names(i, :);
end

for i = 1:length(file{1}.unit_names)
    unit_names{1, i} = file{1}.unit_names(i, :);
end

for i = 1:length(file)
    if ~isempty(file{i}.timeframe)
        EMG = [EMG; file{i}.EMG];
        spike_counts = [spike_counts; file{i}.spike];
        time_frame = [time_frame; file{i}.timeframe.'];
        raw_EMG = [raw_EMG; file{i}.raw_EMG];
        raw_EMG_time_frame = [raw_EMG_time_frame; file{i}.raw_EMG_timeframe.'];
       % FSR_data = [FSR_data; file{i}.FSR_data];
        behavior{i} = file{i}.label;
        behavior_start_time(i) = file{i}.timeframe(1);
        behavior_end_time(i) = file{i}.timeframe(1);
        for j = 1:neurons
            spikes{1, j} = [spikes{1, j}; file{i}.spike_timing{j}.'];
        end
    end
end

xds.time_frame = time_frame;
xds.unit_names = unit_names;
xds.spikes = spikes;
xds.spike_counts = spike_counts;
xds.EMG = EMG;
xds.EMG_names = EMG_names;
xds.raw_EMG = raw_EMG;
xds.raw_EMG_time_frame = raw_EMG_time_frame;
xds.raw_EMG_fs = raw_EMG_fs;
xds.behavior = behavior;
xds.behavior_start_time = behavior_start_time;
xds.behavior_end_time = behavior_end_time;
end

    