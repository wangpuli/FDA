%% function plot_FR_EMG(xds,varargin)
% a helpful_function to set up a heatmap raster for firing rates and plot
% the binnedEMGs in a consistent way that will link the x axes together so
% that zooming on specific sections works well
%
% -- optional inputs --
%  EMG_names - a cell array of the EMGs you want to display. Deault is all
%  'legend_on'/'legend_off' - do we want to display the legend. Default yes
%
% KevinHP 2018
function plot_FR_EMG(xds,varargin)

legend_on = true;
EMG_names = xds.EMG_names;


for ii = 1:nargin-1
    switch class(varargin{ii})
        case 'cell'
            EMG_names = varargin{ii};
        case 'string'
            if strcmpi(varargin{ii},'legend_off')
                legend_on = false;
            end
    end
end



% quick input checks
binFRSize = size(xds.spike_counts); 
binEMGSize = size(xds.EMG);
binTSize = size(xds.time_frame);

% this assumes there are a lot more timepoints than dimensions 
if (max(binFRSize)~=max(binEMGSize))|(max(binFRSize)~=max(binTSize))
    error('The dimensions in the XDS don''t line up. Wattup wit dat?');
end


%% making the firing rate raster
figure
ax(1) = subplot(2,1,1);


% super straight forward for the moment
if binFRSize(1)>binFRSize(2)
    imagesc(xds.spike_counts')
else
    imagesc(xds.spike_counts)
end

ax(1).XTick = [];
zz = zoom;
setAllowAxesZoom(zz,ax(1),false); % make it so we can only zoom in the EMG plot. May change that later
ylabel('Channel')
title('Firing Rates')

%% EMGs
ax(2) = subplot(2,1,2);

[plotEMGs,iPlotEMGs,~] = intersect(xds.EMG_names,EMG_names);
if numel(plotEMGs) < 1
    error('Check the EMG names you supplied')
end

plot(xds.time_frame,xds.EMG(:,iPlotEMGs))
set(ax(2),'XLim',xds.time_frame([1 end]))
% keyboard
lh1 = addlistener(ax(2),'XLim','PostSet',...
    @(src,event) ax2Callback(src,event,ax,xds.bin_width)); % set the axes to match, I guess
% setappdata(ax(2),'XLim_listener',@(src,event) ax2Callback(src,event,ax));
xlabel('Time (s)')
ylabel('EMG Amplitude')
title('EMGs')

if legend_on
    legend(plotEMGs)
end

Leefy


end


%% axis callback function to make zooming easier
function ax2Callback(src,event,ax,binSize)
% keyboard
    set(ax(1),'XLim',ax(2).XLim/binSize);
end