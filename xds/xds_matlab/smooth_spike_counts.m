function [smoothed_spike_counts] = smooth_spike_counts( xds,varargin )

optargin = numel(varargin);
kernel_SD = 0.05;
if optargin > 0
    kernel_SD = varargin{1};
end

if nargin >= 3
   transform = varargin{2}; 
else
   transform = 'sqrt'; 
end
bin_size = xds.time_frame(2) - xds.time_frame(1);

rawSpikeRate = xds.spike_counts;

switch transform
    % sqrt transform of the firing rates
    case 'sqrt'
        spikeRateTransf = sqrt(rawSpikeRate);
    case 'none'
        spikeRateTransf  = rawSpikeRate;
    otherwise
        error('wrong variance stabilization method');
end

disp('Smoothing the firing rates...');

% get nbr of channels and nbr of samples
[nSamples,nCh]  = size(spikeRateTransf);
%[nSamplesEMG,nChEMG]=size(rawEMG);
% preallocate return matrix
smoothedSpikeRate = zeros(nSamples,nCh);
%smoothedEMG=zeros(nSamplesEMG,nChEMG);
% kernel half length is 3??SD out
kernel_hl = ceil( 3 * kernel_SD / (bin_size) );
% create the kernel --it will have length 2*kernel_hl+1
kernel= normpdf( -kernel_hl*(bin_size):bin_size:kernel_hl*(bin_size),0, kernel_SD );
% compute normalization factor --this factor depends on the number of taps
% actually used 
nm = conv(kernel,ones(1,nSamples))';

% do the smoothing
for i = 1:nCh
    aux_smoothed_FR = conv(kernel,spikeRateTransf(:,i)) ./ nm;
    % cut off the edges so that the result of conv is same length as the
    % original data
	smoothed_spike_counts(:,i) = aux_smoothed_FR(kernel_hl+1:end-kernel_hl);
end
end


