function [time_frame,EMG] = resample_EMG(xds, bin_size)

n = bin_size/xds.bin_width;
L = int32(floor(size(xds.EMG, 1)/n));
idx = [];
for i =1:L-1
    idx = [idx, int32(floor(i*n))+1];
end

time_frame = xds.time_frame(idx);
EMG = xds.EMG(idx, :);

end

