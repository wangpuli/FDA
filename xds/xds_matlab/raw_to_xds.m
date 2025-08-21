function xds = raw_to_xds(file_dir, file_name, map_dir, map_name, params)
% raw_to_xds(file_dir, file_name, map_dir, map_name, params)
%
% takes in raw .nev and associated files, and spits out an XDS structure.
% This structure will be built according to the information given by the
% params structure that's input.
%
% Despite the name, this goes through the CDS structure as an intermediary.
% Take that into account - any issues in that codebase will be propagated
% through to this output.
%
% -- Inpusts --
% file_dir              Where the file is stored
% file_name             file name to be converted
% map_dir               directory where the array map is stored
% map_name              name of the map file
% params:        
%   monkey_name         monkey's name 
%   array_name          nickname for the array (ie M1). use the name from
%                           the implant database
%   task_name           valid task code according to the CDS code
%   ran_by              identification of who collected the data
%   lab                 lab #
%   bin_width           width to bin the spiking and EMG
%   sorted              Is the file sorted? (0/1 or true/false)
%   requires_raw_emg    True for all EMG recordings
%   save_waveforms      0 for not saving spike waveforms, 1 for saving
%                       spike waveforms
%
% -- Outputs --
% xds                   The XDS file structure
%



data_file = strcat(file_dir, file_name);
map_file = strcat(map_dir, map_name);
monkey_name = params.monkey_name;
array_name = params.array_name;
task_name = params.task_name;
lab = params.lab;
ran_by = params.ran_by;
sorted = params.sorted;

% Here are some fields for raw data
requires_raw_emg = 0;
requires_raw_force = 0;
if isfield(params,'requires_raw_emg')
    requires_raw_emg = params.requires_raw_emg;
end
if isfield(params,'requires_raw_force')
    requires_raw_force = params.requires_raw_force;
end

% Whether to save the spike waveforms
save_waveforms = 0;
if isfield(params, 'save_waveforms')
    save_waveforms = 1;
end

cds=commonDataStructure();
cds.file2cds(data_file,['array', array_name],...
            ['monkey', monkey_name],lab,'ignoreJumps',['task', task_name], ...
            ['ranBy', ran_by], ['mapFile', map_file]);

bin_width = params.bin_width;
ex = experiment; 
ex.meta.hasEmg = cds.meta.hasEmg; 
ex.meta.hasUnits = true;
ex.meta.hasTrials = true; 
ex.meta.hasForce = cds.meta.hasForce;
ex.meta.hasKinematics = cds.meta.hasKinematics;
ex.meta.hasLfp = cds.meta.hasLfp;
if isempty(cds.kin) && ~isempty(cds.cursor)
    ex.meta.hasKinematics = 1;
end
    
ex.binConfig.filterConfig.sampleRate = 1/bin_width;
ex.binConfig.filterConfig.poles = 2;
ex.binConfig.filterConfig.cutoff = 10;

ex.firingRateConfig.sampleRate = 1/bin_width;
ex.firingRateConfig.method = 'bin';
ex.addSession(cds);
ex.calcFiringRate;

ex.binConfig.include(1).field = 'units';
field_ind = 2;
if ex.meta.hasEmg == true
   ex.emg.processDefault;
   ex.binConfig.include(field_ind).field = 'emg';
   field_ind = field_ind + 1;
end

if ex.meta.hasKinematics == true
   ex.binConfig.include(field_ind).field = 'kin';
   field_ind = field_ind + 1;
end

if ex.meta.hasForce == true
   ex.binConfig.include(field_ind).field = 'force';
end

% if ex.meta.hasLfp == true
%    ex.binConfig.include(field_ind).field = 'lfp';
% end

%^The above block of text is sometimes necessary if you're converting files
%from monkeys that have multiple implants. Such a file can have only
%units and LFP with no EMG/kinematics/cursor data, because the
%EMG/kinematics/cursor data is in the file recorded from the other array;
%trying to create an XDS with only units causes the experiment processing to
%fail, and so that requires the inclusion of LFP in the experiment processing. 
ex.binData;

xds.meta = cds.meta;
xds.bin_width = bin_width;
xds.time_frame = ex.bin.data.t;
xds.has_EMG = xds.meta.hasEmg;
xds.has_force = xds.meta.hasForce;
xds.has_kin = xds.meta.hasKinematics;
xds.sorted = sorted;

% units
% In some data files (like those collected on Monkey Chewie in 2015) there
% are data entries in the unit table but they are not the actual neural
% data. In order to find them out and get rid of them later, the next lines
% of code do a quick scan over the labels. If the labels are initialed with
% 'elec', then the corresponding channel is neural data, and will be
% registered in 'elec_mask'
% Find out invalid channels based on ID number
invalid_id = find([cds.units(:).ID] == 255);
% Get rid of the invalid channels in the cds.unit table and get a copy
temp_table = cds.units;
temp_table(invalid_id) = [];
    
elec_mask = zeros(length(temp_table), 1);
for i = 1:length(temp_table)
    if strfind(temp_table(i).label,'elec') == 1
        elec_mask(i) = 1;
    end
end

if sorted == 0
    good_id = find(elec_mask == 1);
    for i = 1:length(good_id)
        xds.unit_names{1,i} = temp_table(good_id(i)).label;
        xds.spikes{1,i} = temp_table(good_id(i)).spikes.ts;
        if save_waveforms == 1
           xds.spike_waveforms{1,i} = temp_table(good_id(i)).spikes.wave;
        end
    end
    % Finding out the column numbers for the firing rates in the table of
    % ex.bin.data
    [~,binnedUnitMask] = ex.bin.getUnitNames;
    % Finding out the positions of the '0's in elec_mask
    bad_id = find(elec_mask == 0);
    temp1 = find(binnedUnitMask == 1);
    % Setting the masks for the fake channels to '0'
    for i = 1:length(bad_id)
        binnedUnitMask(temp1(bad_id(i))) = 0;
    end
elseif sorted == 1
    good_id = find(elec_mask == 1);
    k = 1;
    for i = 1:length(good_id)
        if (temp_table(i).ID ~= 0)&(temp_table(i).ID ~= 255)
            xds.unit_names{1, k} = strcat(temp_table(good_id(i)).label,...
                strcat('_', char(string(temp_table(i).ID))) );
            xds.spikes{1, k} = temp_table(good_id(i)).spikes.ts;
            if save_waveforms == 1
               xds.spike_waveforms{1,k} = temp_table(good_id(i)).spikes.wave;
            end
            k = k + 1;
        end 
    end
    
    [~,binnedUnitMask] = ex.bin.getUnitNames;
    
    bad_id = find(elec_mask == 0);
    nonsorted_id = find([temp_table(:).ID] == 0);
    
    temp1 = find(binnedUnitMask == 1);
    for i = 1:length(bad_id)
        binnedUnitMask(temp1(bad_id(i))) = 0;
    end 
    for i = 1:length(nonsorted_id)
        binnedUnitMask(temp1(nonsorted_id(i))) = 0;
    end
end

xds.spike_counts = ex.bin.data{:,binnedUnitMask}*bin_width;
if ex.meta.hasEmg == true
   emgMask = ~cellfun(@(x)isempty(strfind(x,'EMG')),ex.bin.data.Properties.VariableNames);
   emgNames = ex.bin.data.Properties.VariableNames(emgMask);
   xds.EMG = ex.bin.data{:,emgMask};
   xds.EMG_names = emgNames;
   if requires_raw_emg
       raw_EMG_table = table2array(cds.emg);
       xds.raw_EMG = raw_EMG_table(:,2:end);
       xds.raw_EMG_time_frame = raw_EMG_table(:,1);
   else
       xds.raw_EMG = [];
       xds.raw_EMG_time_frame = [];
   end
end
if ex.meta.hasForce == true
   fxMask = ~cellfun(@(x)isempty(strfind(x,'fx')),ex.bin.data.Properties.VariableNames);
   fyMask = ~cellfun(@(x)isempty(strfind(x,'fy')),ex.bin.data.Properties.VariableNames);
   xds.force(:, 1) = ex.bin.data{:,fxMask};
   xds.force(:, 2) = ex.bin.data{:,fyMask};
   if requires_raw_force
       raw_force_table = table2array(cds.force);
       xds.raw_force = raw_force_table(:,2:end);
       xds.raw_force_time_frame = raw_force_table(:,1);
   else
       xds.raw_force = [];
       xds.raw_force_time_frame = [];
   end
end
if ex.meta.hasKinematics == true
   xMask = ~cellfun(@(x)isempty(strfind(x,'x')),ex.bin.data.Properties.VariableNames);
   yMask = ~cellfun(@(x)isempty(strfind(x,'y')),ex.bin.data.Properties.VariableNames);
   temp = find(xMask==1);
   for i = 2:length(temp)
       xMask(temp(i)) = 0; 
   end
   temp = find(yMask==1);
   for i = 2:length(temp)
       yMask(temp(i)) = 0; 
   end
   vxMask = ~cellfun(@(x)isempty(strfind(x,'vx')),ex.bin.data.Properties.VariableNames);
   vyMask = ~cellfun(@(x)isempty(strfind(x,'vy')),ex.bin.data.Properties.VariableNames);
   axMask = ~cellfun(@(x)isempty(strfind(x,'ax')),ex.bin.data.Properties.VariableNames);
   ayMask = ~cellfun(@(x)isempty(strfind(x,'ay')),ex.bin.data.Properties.VariableNames);
   xds.curs_p(:, 1) = ex.bin.data{:, xMask};
   xds.curs_p(:, 2) = ex.bin.data{:, yMask};
   xds.curs_v(:, 1) = ex.bin.data{:, vxMask};
   xds.curs_v(:, 2) = ex.bin.data{:, vyMask};
   xds.curs_a(:, 1) = ex.bin.data{:, axMask};
   xds.curs_a(:, 2) = ex.bin.data{:, ayMask};
end   

% trial information
xds.trial_info_table_header = fieldnames(cds.trials);
xds.trial_info_table = table2cell(cds.trials);

xds.trial_gocue_time = deal_trial_info('goCue', cds);
xds.trial_start_time = deal_trial_info('startTime', cds);
xds.trial_end_time = deal_trial_info('endTime', cds);
xds.trial_result = deal_trial_info('result', cds);
xds.trial_target_dir = deal_trial_info('tgtDir', cds);
xds.trial_target_corners = deal_trial_info('Corners', cds);

clear cds
clear ex
end

function trial_info = deal_trial_info(str,cds)
trial_mask = ~cellfun(@(x)isempty(strfind(x,str)),cds.trials.Properties.VariableNames);
if sum(trial_mask) == 0
    disp('Something is wrong with the trial table');
    trial_info = 0;
else 
    trial_info = cds.trials{:, trial_mask};
end
end