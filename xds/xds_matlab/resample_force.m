function [time_frame,force] = resample_force(xds, bin_size)

n = bin_size/xds.bin_width;
L = int32(floor(size(xds.force, 1)/n));
idx = [];
for i =1:L-1
    idx = [idx, int32(floor(i*n))+1];
end

time_frame = xds.time_frame(idx);
force = xds.force(idx, :);

end

