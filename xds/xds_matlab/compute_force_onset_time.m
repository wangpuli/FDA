function time_onset = compute_force_onset_time(xds)
% The force onset time during all trials, including rewarded, failed and 
% aborted trials are calculated here

idx = cell(length(xds.trial_start_time), 1);
for i = 1:length(idx)
    idx{i} = find((xds.time_frame > xds.trial_start_time(i)) & (xds.time_frame < xds.trial_end_time(i)));
end

trial_time_frame = cell(length(idx), 1);
for i = 1:length(idx)
    trial_time_frame{i} = xds.time_frame(idx{i});
end

trial_force = cell(length(idx), 1);
for i = 1:length(idx)
    trial_force{i} = xds.force(idx{i}, :);
end

thr = 0.4;
idx_onset = find_force_onset(trial_force, thr);

time_onset = [];
for i = 1:length(idx_onset)
    time_onset = [time_onset trial_time_frame{i}(idx_onset(i))];
end

time_onset = time_onset';

disp('Force onset time computed')

end

