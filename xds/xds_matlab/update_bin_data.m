function xds = update_bin_data(xds, new_bin_size)

if isfield(xds, 'has_cursor')
    flag_cursor = xds.has_cursor;
elseif isfield(xds, 'has_kin')
    flag_cursor = xds.has_kin;
end
flag_cursor = 1;

[t_spike, spike_counts] = bin_spikes(xds, new_bin_size);
if xds.has_EMG
    [t_EMG, EMG] = resample_EMG(xds, new_bin_size);
    if length(t_EMG) > length(t_spike)
        EMG = EMG(1:length(t_spike), :);
    end
end
if xds.has_force
    [t_force, force] = resample_force(xds, new_bin_size);
    if length(t_force) > length(t_spike)
        force = force(1:length(t_spike), :);
    end
end
if flag_cursor
    [t_curs, curs_p, curs_v, curs_a] = resample_kin(xds, new_bin_size);
    if length(t_curs) > length(t_spike)
        curs_p = curs_p(1:length(t_spike), :);
        curs_v = curs_p(1:length(t_spike), :);
        curs_a = curs_p(1:length(t_spike), :);
    end
end

xds.time_frame = t_spike;
xds.bin_width = new_bin_size;
xds.spike_counts = spike_counts;
if xds.has_EMG
    xds.EMG = EMG;
end
if xds.has_force
    xds.force = force;
end
if flag_cursor
    if isfield(xds, 'has_kin')
        xds.kin_p = curs_p;
        xds.kin_v = curs_v;
        xds.kin_a = curs_a;
    elseif isfield(xds, 'has_cursor')
        xds.curs_p = curs_p;
        xds.curs_v = curs_v;
        xds.curs_a = curs_a;

    end

end

end




