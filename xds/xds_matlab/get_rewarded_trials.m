function [trial_spike_counts, trial_EMG, trial_force, trial_kin, trial_curs, trial_tgt_pos] = get_rewarded_trials(xds, start_event, varargin)

if nargin > 3
    time_before_start = varargin{1};
    end_event = varargin{2};
    time_after_end = varargin{3};
else 
    time_before_start = 0;
    end_event = 'end_time';
    time_after_end = 0;
end


% a list of times for successes
suc_timetable = get_trial_time_table(xds, 'R', start_event, time_before_start, end_event, time_after_end);

% set a sub-index for incrementing the cell when it meets all of the
% conditions we want
j = 1;

% Trial target positions
trial_tgt_pos = [];
% trial_tgt_pos_succ = xds.trial_target_dir(xds.trial_result == 'R');
trial_tgt_dir_succ = xds.trial_target_dir(xds.trial_result == 'R');

% I think we should be able to change this into just logical indexing.
for i = 1:length(suc_timetable)
    
    % if the length of the trial is less than 500 ms, skip
    if suc_timetable(i,2) - suc_timetable(i,1)<0.5
        continue;
    end
    
    % if either of the timestamps are NaN
    if isnan(suc_timetable(i,1)) || isnan(suc_timetable(i,2))
       continue;
    end
    
    % find the timestamps that particular trial
    trial_inds = find((xds.time_frame >= suc_timetable(i,1))&(xds.time_frame <= suc_timetable(i,2)));
    trial_spike_counts{j,1} = xds.spike_counts(trial_inds, :);
    
    % fill in EMGs if we have it
    if xds.has_EMG == true
       trial_EMG{j,1} = xds.EMG(trial_inds, :);
    else
       trial_EMG = 0;
    end
    
    % fill in forces if we have it
    if xds.has_force == true
       trial_force{j,1} = xds.force(trial_inds, :);
    else
       trial_force = 0;
    end
    
    % kinematics of the robot handle if we have them
    if xds.has_kin == true
       trial_kin{j,1} = xds.kin_p(trial_inds, :);
       trial_kin{j,2} = xds.kin_v(trial_inds, :);
       trial_kin{j,3} = xds.kin_a(trial_inds, :);
    else
       trial_kin = 0;
    end
    
    % cursor positions (we better have them!)
    if ~isempty(xds.curs_p)
        trial_curs{j,1} = xds.curs_p(trial_inds, :); % cursor position
        trial_curs{j,2} = xds.curs_v(trial_inds, :); % cursor velocity
        trial_curs{j,3} = xds.curs_a(trial_inds, :); % cursor acceleration
    else
        trial_curs = 0;
    end
    
    % better also have this!
    trial_tgt_pos = [trial_tgt_pos; trial_tgt_dir_succ(i)];
    j = j+1;
end

end

