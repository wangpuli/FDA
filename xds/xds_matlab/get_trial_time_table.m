function [trial_time_table] = get_trial_time_table(xds, trial_type, start_event, varargin)
% Function that returns trial start and end time according to the chosen
% start and end event. By default, end event is end_time. Can specify time
% before start and after end as well (0 by default).
%
% Usage: get_trial_time_table(xds, 'R', 'gocue_time')
%        extract time for rewarded trials starting at gocue
% 
% Usage: get_trial_time_table(xds, 'R', 'force_onset_time', 0.5, 'force_osnet_time, 1)
%        extract time for rewraded trials starting 0.5s before force onset
%        and ending 1s after force onset

if nargin > 3
    time_before_start = varargin{1};
    end_event = varargin{2};
    time_after_end = varargin{3};
else 
    time_before_start = 0;
    end_event = 'end_time';
    time_after_end = 0;
end

% For trial_type, 'R': rewarded trials, 'A': aborted trials, 'F': failed
temp = find(xds.trial_result == trial_type);
trial_time_table = zeros(length(temp),2);

for i =1:size(temp,1)
    idx = temp(i);
    if strcmp(start_event,'start_time')
        time_start = xds.trial_start_time(idx);
    elseif strcmp(start_event,'gocue_time')
        time_start = xds.trial_gocue_time(idx);
    elseif strcmp(start_event,'force_onset_time')
        time_start = xds.trial_force_onset_time(idx);
    elseif strcmp(start_event,'end_time')
        time_start = xds.trial_end_time(idx);
    end

    if strcmp(end_event,'start_time')
        time_end = xds.trial_start_time(idx);
    elseif strcmp(end_event,'gocue_time')
        time_end = xds.trial_gocue_time(idx);
    elseif strcmp(end_event,'force_onset_time')
        time_end = xds.trial_force_onset_time(idx);
    elseif strcmp(end_event,'end_time')
        time_end = xds.trial_end_time(idx);
    end

    t1 = time_start - time_before_start;
    t2 = time_end + time_after_end;

    if t2 > t1
        trial_time_table(i,:) = [t1 t2];
    else
        trial_time_table(i,:) = [];
%         print('The timing with trial No. %d is not right.'%(n))
    end
end
end

