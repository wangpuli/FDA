function onset_num = find_force_onset(force_list, thr)
% This function is designed to find out the force onset time for each trial
% force_list: a list of force signals, each element corresponds to a trial
% thr: the threshold to determine whether the force is on

onset_num = zeros(length(force_list), 1);
for i = 1:length(force_list)
    each = force_list{i};
    f = sqrt(each(:, 1).^2 + each(:, 2).^2);
    df = diff(f);
    temp = find(df >= thr*max(df));
       
    if isempty(temp)
        onset_num(i) = 1;
    else
        onset_num(i) = temp(1);
    end
end

end

