function [time_frame,curs_p,curs_v,curs_a] = resample_kin(xds, bin_size)

n = bin_size/xds.bin_width;
if isfield(xds, 'kin_p')
    L = int32(floor(size(xds.kin_p, 1)/n));
else
    L = int32(floor(size(xds.curs_p, 1)/n));
end

idx = [];
for i =1:L-1
    idx = [idx, int32(floor(i*n))+1];
end

time_frame = xds.time_frame(idx);
if isfield(xds, 'kin_p')
    curs_p = xds.kin_p(idx, :);
    curs_v = xds.kin_v(idx, :);
    curs_a = xds.kin_a(idx, :);
else
    curs_p = xds.curs_p(idx, :);
    curs_v = xds.curs_v(idx, :);
    curs_a = xds.curs_a(idx, :);
end

end

