function [time_frame,spike_counts] = bin_spikes(xds, bin_size, varargin)

optargin = numel(varargin);
mode = 'center';
if optargin > 0
    mode = 'left';
end
disp(['The new bin size is ', num2str(bin_size), ' s']) 
spike_counts = [];       
if mode == 'center'
    bins = xds.time_frame(1) - bin_size/2:bin_size:xds.time_frame(end) + bin_size/2;
elseif mode == 'left'
    bins = xds.time_frame(1):bin_size:xds.time_frame(end);
end
for i=1:length(xds.spikes)
    bb = xds.spikes{i};
    [out,edges] = histcounts(bb,bins);
    spike_counts = [spike_counts, out'];
end

time_frame = bins(2:end)';

end

