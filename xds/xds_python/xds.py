import sys
import numpy as np
import scipy.io as sio
from scipy import stats, signal
from load_intan_rhd_format import read_data
import copy
import os.path
from os import path
from collections import defaultdict
from xds_utils import parse_h5py, parse_scipy
from xds_utils import get_char_pos, get_paired_EMG_index, find_bad_EMG_index_from_list, delete_paired_bad_channel
from xds_utils import find_force_onset, find_movement_onset
from xds_utils import find_target_dir

if sys.version[0] == '2':
    import cPickle as pickle
else:
    import _pickle as pickle
    
    
class lab_data:
    def __init__(self, base_path, file_name):
        if base_path[-1] != '/':
            base_path = base_path + '/' 
        self.file_name = file_name[:-4]
        if not path.exists( base_path + file_name ):
            raise Exception( 'Can''t find file:' + file_name )
        self.parse_file(base_path, file_name)
        
    def parse_file(self, base_path, file_name):
        try:
            parsed = parse_scipy(base_path, file_name)
        except Exception:
            parsed = parse_h5py(base_path, file_name)        
        # -------- time_frame -------- #
        self.time_frame = parsed['time_frame']
        # -------- meta -------- #
        self.__meta = {}
        self.__meta['monkey_name'] = parsed['meta']['monkey']
        self.__meta['task_name'] = parsed['meta']['task']
        self.__meta['duration'] = parsed['meta']['duration']
        self.__meta['collect_date'] = parsed['meta']['dateTime']
        self.__meta['raw_file_name'] = parsed['meta']['rawFileName']
        self.__meta['array'] = parsed['meta']['array']
        self.__meta['processed_time'] = parsed['meta']['processedTime']
        # -------- flag variables -------- #
        self.sorted = parsed['sorted']
        self.bin_width = parsed['bin_width']
        self.has_EMG = parsed['has_EMG']
        self.has_kin = 0
        if 'has_kin' in parsed.keys(): 
            self.has_kin = parsed['has_kin']
        if 'has_cursor' in parsed.keys():
            self.has_cursor = parsed['has_cursor']
        self.has_force = parsed['has_force']
        if 'has_raw_force' in parsed.keys():
            self.has_raw_force = parsed['has_raw_force']
        if 'has_raw_EMG' in parsed.keys():
            self.has_raw_EMG = parsed['has_raw_EMG']
        # -------- trial information -------- #
        self.trial_target_corners = parsed['trial_target_corners']
        self.trial_target_dir = parsed['trial_target_dir'] 
        self.trial_result = parsed['trial_result']
        self.trial_start_time = parsed['trial_start_time']
        self.trial_end_time = parsed['trial_end_time']
        self.trial_gocue_time = parsed['trial_gocue_time']
        self.trial_info_table_header = parsed['trial_info_table_header']
        self.trial_info_table = parsed['trial_info_table']
        # -------- data -------- #
        self.spike_counts = parsed['spike_counts']
        self.spikes = parsed['spikes']
        self.unit_names = parsed['unit_names']
        if 'spike_waveforms' in parsed.keys():
            self.spike_waveforms = parsed['spike_waveforms']
        if 'EMG' in parsed.keys():
            self.EMG = parsed['EMG']
            self.EMG_names = parsed['EMG_names']
        if 'raw_EMG' in parsed.keys():
            self.raw_EMG = parsed['raw_EMG']
            self.raw_EMG_time_frame = parsed['raw_EMG_time_frame']
        if 'force' in parsed.keys():
            self.force = parsed['force']
        if 'raw_force' in parsed.keys():
            self.raw_force = parsed['raw_force']
            self.raw_force_time_frame = parsed['raw_force_time_frame']
        if 'curs_p' in parsed.keys():
            self.curs_p = parsed['curs_p']
            self.curs_v = parsed['curs_v']
            self.curs_a = parsed['curs_a']

        self.n_neural = np.size(self.spike_counts, 1)
        if self.has_EMG == 1:
            self.n_EMG = np.size(self.EMG, 1)
        else:
            self.n_EMG = 0
        if self.has_force == 1:
            try:
                self.n_force = np.size(self.force, 1)
            except:
                pass       
        else:
            self.n_force = 0
            
        # -------- get rid of trials with nan timings -------- #
        self.clean_up_trials()
        
        # -------- find out the target centers -------- #
        try:
            idx = self.trial_info_table_header.index('tgtCtr')
        except Exception:
            try:
                idx = self.trial_info_table_header.index('tgtCenter')
            except Exception:
                print('Check the trial info table header')
        target_center = np.asarray(self.trial_info_table[idx]).squeeze()
        self.trial_target_center_x = target_center[:, 0]
        self.trial_target_center_y = target_center[:, 1]
        
        # -------- for multigadget files with more than 1 gadget activated, find out the gadget number --------- #
        if '_MG_' in self.file_name:
            try:
                idx = self.trial_info_table_header.index('gadgetNumber')
                self.trial_gadget_number = np.asarray(self.trial_info_table[idx]).squeeze()
            except Exception:
                print('Lack of gadget information in the data file!')
            
    def get_meta(self):
        a = dict()
        a = self.__meta
        return a
        
    def print_file_info(self):
        print('Monkey: %s' % (self.__meta['monkey_name']))
        print('Task: %s' % (self.__meta['task_name']))
        print('Collected on %s ' % (self.__meta['collect_date']))
        print('Raw file name is %s' % (self.__meta['raw_file_name']))
        print('The array is in %s' % (self.__meta['array']))
        print('There are %d neural channels' % (self.n_neural))
        print('Sorted? %d' % (self.sorted))
        print('There are %d EMG channels' % (self.n_EMG))
        print('Current bin width is %.4f seconds' % (self.bin_width))
        if self.has_EMG == 1:
            print('The name of each EMG channel:')
            for i in range(len(self.EMG_names)):
                print(self.EMG_names[i])
        print('The dataset lasts %.4f seconds' % (self.__meta['duration']))
        print('There are %d trials' % (len(self.trial_result)))
        print('In %d trials the monkey got reward' % (len(np.where(self.trial_result == 'R')[0])))
    
    def print_trial_info_table_header(self):
        for each in self.trial_info_table_header:
            print(each)
            
    def get_one_colum_in_trial_info_table(self, colum_name):
        n = np.where(np.asarray(self.trial_info_table_header) == colum_name)[0][0]
        a = [each[n][0][0] for each in self.trial_info_table]
        return a
    
    def save_to_pickle(self, path, file_name = 0):
        if path[-1] != '/':
            path = path + '/'
        if file_name == 0:
            f = path + self.file_name + '.pkl'
        else:
            f = path + file_name + '.pkl'
        with open (f, 'wb') as fp:
            pickle.dump(self, fp)
        print('Save to %s successfully' %(f))
        
    def clean_up_trials(self):
        """
        For some trials the timings for trial_start, trial_end or trial_gocue are nans. 
        This function will get rid of these trials.

        Returns
        -------
        None.

        """
        trial_gocue_time = self.trial_gocue_time
        trial_start_time = self.trial_start_time
        trial_end_time = self.trial_end_time
        # -------- Get rid of trials with gocue time nan -------- #
        gocue_nan_idx = np.argwhere(np.isnan(trial_gocue_time))[:,0]
        start_nan_idx = np.argwhere(np.isnan(trial_start_time))[:,0]
        end_nan_idx = np.argwhere(np.isnan(trial_end_time))[:,0]
        union_nan_idx = np.asarray(sorted(list(set(gocue_nan_idx).union(set(start_nan_idx)).union(set(end_nan_idx))),
                                          reverse = True))
        if len(union_nan_idx)>0:
            self.trial_gocue_time = np.delete(self.trial_gocue_time, union_nan_idx, axis = 0)
            self.trial_start_time = np.delete(self.trial_start_time, union_nan_idx)
            self.trial_end_time = np.delete(self.trial_end_time, union_nan_idx)
            self.trial_result = np.delete(self.trial_result, union_nan_idx)
            try:
                self.trial_target_dir = np.delete(self.trial_target_dir, union_nan_idx)
            except Exception:
                print('Target direction not applicable')
            try:
                self.trial_target_corners = np.delete(self.trial_target_corners, union_nan_idx, axis = 0)
            except Exception:
                print('Target corners not applicable')
            for each in self.trial_info_table:
                for idx in union_nan_idx:
                    del(each[idx])
            print('Trials with nan timings have been removed!')
        
    def compute_force_onset_time(self, channel = 0):
        """
        The force onset time during all trials, including rewarded, failed and aborted trials
        are calculated here
        """
        if hasattr(self, 'force'):
            idx = [np.where((self.time_frame > t[0]) & (self.time_frame < t[1]) )[0] 
                   for t in zip(self.trial_start_time, self.trial_end_time)]
            trial_time_frame = [self.time_frame[n] for n in idx]
            trial_force = [self.force[n] for n in idx]
            idx_onset = find_force_onset(trial_force, channel, 0.4)
            time_onset = [trial_time_frame[i][idx_onset[i]] for i in range(len(trial_time_frame))]
            print('Get the force onset time!')
            self.trial_force_onset_time = np.asarray(time_onset)
        else:
            print('There is no force data in this file')
    
    def compute_movement_onset_time(self, channel = 0, thr = 0.4):
        """
        This function is almost the same as the one defined above. For consistency considerations
        here both functions are kept.
        """
        if (hasattr(self, 'curs_p')|hasattr(self, 'kin_p')):
            idx = [np.where((self.time_frame > t[0]) & (self.time_frame < t[1]) )[0] 
                   for t in zip(self.trial_start_time, self.trial_end_time)]
            trial_time_frame = [self.time_frame[n] for n in idx]
            trial_curs_p = [self.curs_p[n] for n in idx]
            idx_onset = find_movement_onset(trial_curs_p, channel, thr)
            time_onset = [trial_time_frame[i][idx_onset[i]] for i in range(len(trial_time_frame))]
            print('Get the movement onset time!')
            self.trial_movement_onset_time = np.asarray(time_onset)
        else:
            print('There is no force data in this file')
    
    def get_trials_idx(self, my_type, start_event, time_before_start, end_event = 'end_time', time_after_end = 0, raw_flag = 0, gadget_number = -1):
        """
        This function returns a list containing the indices for extracting the data for each trial

        Parameters
        ----------
        my_type : A char in uppercase
            Specifying the type of the trial, 'R' for successful trials, 'F' for failed trials, 'ALL' for all trials.
        start_event : A string
            Specifying the start event for each trial, including 'start_time', 'gocue_time' and 'end_time'.
        time_before_start : float
            Specifying the time before the trial_start event.
        end_event : A string
            Specifying the end event for each trial. The default is 'end_time'.
        time_after_end : float, optional
            The time after the trial_end event. The default is 0.
        gadget_number: bool, optional
            The number of the gadget in Multi-gadget (MG) task, typically used for discriminate PG and KEY trials.

        Returns
        -------
        trials_idx : list
            A list containing the indices for each trial.

        """
        if start_event == 'start_time':
            time_start = self.trial_start_time
        elif start_event == 'gocue_time':
            time_start = self.trial_gocue_time
        elif start_event == 'end_time':
            time_start = self.trial_end_time
        elif start_event == 'force_onset_time':
            try:
                time_start = self.trial_force_onset_time
            except Exception:
                print('Compute force onset time first')
        elif start_event == 'movement_onset_time':
            try:
                time_start = self.trial_movement_onset_time
            except Exception:
                print('Compute movement onset time first')
            
        if end_event == 'start_time':
            time_end = self.trial_start_time
        elif end_event == 'gocue_time':
            time_end = self.trial_gocue_time
        elif end_event == 'end_time':
            time_end = self.trial_end_time
        elif end_event == 'force_onset_time':
            try:
                time_end = self.trial_force_onset_time
            except Exception:
                print('Compute force onset time first')
        elif end_event == 'movement_onset_time':
            try:
                time_end = self.trial_movement_onset_time
            except Exception:
                print('Compute movement onset time first')
        
        if (my_type == 'R')|(my_type == 'F'):
            # Get the indices of a specific type of trials, 'R' or 'F'
            type_trial = np.where(self.trial_result == my_type)[0]
        elif my_type == 'ALL':
            type_trial = np.arange( len(self.trial_result) )
        # If gadget number is something, not -1, then do this:
        if gadget_number != -1:
            # The number of trials with a specific gadget number
            gadget_trial = np.where(self.trial_gadget_number == gadget_number)[0]
            temp = sorted(list(set(type_trial)&set(gadget_trial)))
            type_trial = temp
        trials_idx = []
        if len(type_trial) > 0:
            for n in type_trial:
                # RT go_cue
                t1 = time_start[n] - time_before_start
                t2 = time_end[n] + time_after_end
                if isinstance(t1, np.ndarray):
                    t1 = t1[-1]
                if t2 > t1:
                    if raw_flag == 0:
                        idx = np.where( (self.time_frame > t1) & (self.time_frame <= t2) )[0]
                    else:
                        if hasattr(self, 'raw_EMG_time_frame'):
                            idx = np.where( (self.raw_EMG_time_frame > t1) & (self.raw_EMG_time_frame <= t2) )[0]
                        else:
                            #print('There is no raw EMG in this file, but the raw_flag is 1 here, please check')
                            idx = []
                else:
                    idx = []
                    print('The timing with trial No. %d is not right.'%(n))
                trials_idx.append(idx)
        return trials_idx
    
    def get_trial_info(self, my_type = 'R', gadget_number = -1):
        trial_info_list = []
        if (my_type == 'R')|(my_type == 'F'):
            type_trial = np.where(self.trial_result == my_type)[0]
        elif my_type == 'ALL':
            type_trial = np.arange( len(self.trial_result) )
        if gadget_number != -1:
            # The number of trials with a specific gadget number
            gadget_trial = np.where(self.trial_gadget_number == gadget_number)[0]
            temp = sorted(list(set(type_trial)&set(gadget_trial)))
            type_trial = temp
        if len(type_trial)>0:
            for each in type_trial:
                trial_info = {}
                trial_info['trial_result'] = self.trial_result[each]
                try:
                    trial_info['trial_target_dir'] = self.trial_target_dir[each]
                except Exception:
                    pass #print('Target directions not applicable')
                trial_info['trial_gocue_time'] = self.trial_gocue_time[each]
                trial_info['trial_start_time'] = self.trial_start_time[each]
                trial_info['trial_end_time'] = self.trial_end_time[each]
                try:
                    trial_info['trial_target_corners'] = self.trial_target_corners[each, :]
                except Exception:
                    pass #print('Target corners not applicable')
                if hasattr(self, 'trial_force_onset_time'):
                    trial_info['trial_force_onset_time'] = self.trial_force_onset_time[each]
                if hasattr(self, 'trial_target_center_x'):
                    trial_info['trial_target_center_x'] = self.trial_target_center_x[each]
                if hasattr(self, 'trial_target_center_y'):
                    trial_info['trial_target_center_y'] = self.trial_target_center_y[each]
                if hasattr(self, 'trial_gadget_number'):
                    trial_info['trial_gadget_number'] = self.trial_gadget_number[each]
                if hasattr(self, 'trial_movement_onset_time'):
                    trial_info['trial_movement_onset_time'] = self.trial_movement_onset_time[each]
                trial_info_list.append(trial_info)
        return trial_info_list         
    
    def get_trials_data_spike_counts(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                  end_time_offset, gadget_number = gadget_number)
        trial_spike_counts = [self.spike_counts[n, :] for n in idx]
        return trial_spike_counts
    
    def get_trials_data_time_frame(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                  end_time_offset, gadget_number = gadget_number)
        return [self.time_frame[n] for n in idx]
    
    def get_trials_data_EMG(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, EMG_channels = 'all', gadget_number = -1):
        if self.has_EMG == 0:
            print('There is no EMG in this file')
            return 0
        else:
            idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                      end_time_offset, gadget_number = gadget_number)
            if EMG_channels == 'all':
                return [self.EMG[n, :] for n in idx]
            else:
               EMG_channels_idx = [self.EMG_names.index(each) for each in EMG_channels]
               temp = self.EMG[:, EMG_channels_idx]
               return [temp[n, :] for n in idx]
        
    def get_trials_data_raw_EMG(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, EMG_channels = 'all', gadget_number = -1):
        if self.has_EMG == 0:
            print('There is no EMG in this file')
            return 0
        else:
            idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                      end_time_offset, 1, gadget_number = gadget_number)
            if EMG_channels == 'all':
                return [self.raw_EMG[n, :] for n in idx]
            else:
               EMG_channels_idx = [self.EMG_names.index(each) for each in EMG_channels] 
               return [self.raw_EMG[n, EMG_channels_idx] for n in idx]

    def get_trials_data_force(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        if self.has_force == 0:
            print('There is no force in this file')
            return 0
        else:
            idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                      end_time_offset, gadget_number = gadget_number)
            return [self.force[n, :] for n in idx]
    
    def get_trials_data_target_center(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                  end_time_offset, gadget_number = gadget_number)
        return [[self.trial_target_center_x[n], self.trial_target_center_y[n]] for n in idx]

    def get_trials_data_kin(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        if hasattr(self, 'has_cursor'):
            flag = self.has_cursor
        elif hasattr(self, 'has_kin'):
            flag = self.has_kin
        
        if flag == 0:
            print('There is no cursor trajectories in this file!')
            return 0
        else:
            idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                      end_time_offset, gadget_number = gadget_number)
            if hasattr(self, 'curs_p'):
                return [self.curs_p[n, :] for n in idx], [self.curs_v[n, :] for n in idx], [self.curs_a[n, :] for n in idx]
            elif hasattr(self, 'kin_p'):
                return [self.kin_p[n, :] for n in idx], [self.kin_v[n, :] for n in idx], [self.kin_a[n, :] for n in idx]
    
    def get_trials_data_cursor(self, my_type = 'R', trial_start = 'start_time', time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        if hasattr(self, 'has_cursor'):
            flag = self.has_cursor
        elif hasattr(self, 'has_kin'):
            flag = self.has_kin
        
        if flag == 0:
            print('There is no cursor trajectories in this file!')
            return 0
        else:
            idx = self.get_trials_idx(my_type, trial_start, time_ahead, end_event, 
                                      end_time_offset, gadget_number = gadget_number)
            if hasattr(self, 'curs_p'):
                return [self.curs_p[n, :] for n in idx], [self.curs_v[n, :] for n in idx], [self.curs_a[n, :] for n in idx]
            elif hasattr(self, 'kin_p'):
                return [self.kin_p[n, :] for n in idx], [self.kin_v[n, :] for n in idx], [self.kin_a[n, :] for n in idx]
      
    def get_trials_data_spikes(self, my_type = 'R', trial_start = 'start_time', 
                          time_ahead = 0, end_event = 'end_time', end_time_offset = 0, gadget_number = -1):
        trial_spike = []
        trial_time_frame = self.get_trials_data_time_frame(my_type, trial_start, time_ahead, end_event, end_time_offset, gadget_number)
        for i, t in enumerate(trial_time_frame):
            temp = []
            for j, spike in enumerate(self.spikes):
                idx = np.where( (spike>t[0])&(spike<t[-1]) )[0]
                if len(idx)>0:
                    s = (spike[idx] - t[0]).reshape((spike[idx].shape[0], ))
                else:
                    s = spike[idx].reshape((spike[idx].shape[0], ))
                temp.append(s)
            trial_spike.append(temp)
        return trial_spike
    
    def update_target_dir(self):
        """
        In some files the directions for the reaching targets are wrong. To get the right target directions
        the cursor trajectories are needed.
        By calling this function the target directions will be updated for all trials.
        """
        if hasattr(self, 'curs_p'):
            # trial_curs_p = self.get_trials_data_cursor('ALL', 'gocue_time', 0, 'end_time', 0)[0]
            trial_curs_p = self.get_trials_data_cursor('R', 'gocue_time', 0, 'end_time', 0)[0]
        elif hasattr(self, 'kin_p'):
            # trial_curs_p = self.get_trials_data_kin('ALL', 'gocue_time', 0, 'end_time', 0)[0]
            trial_curs_p = self.get_trials_data_kin('R', 'gocue_time', 0, 'end_time', 0)[0]
        new_dir = find_target_dir(trial_curs_p, [-135, -90, -45, 0, 45, 90, 135, 180])     
        self.trial_target_dir = new_dir
    
    def update_bin_data(self, new_bin_size, update = 1):
        if hasattr(self, 'has_cursor'):
            flag_cursor = self.has_cursor
        elif hasattr(self, 'has_kin'):
            flag_cursor = self.has_kin
        t_spike, spike_counts = self.bin_spikes(new_bin_size)
        if self.has_EMG == 1:
            t_EMG, EMG = self.resample_EMG(new_bin_size)
            if len(t_EMG) > len(t_spike):
                EMG = EMG[:len(t_spike), :]
        if self.has_force == 1:
            t_force, force = self.resample_force(new_bin_size)
            if len(t_force) > len(t_spike):
                force = force[:len(t_spike), :]
        if flag_cursor == 1:
            t_curs, curs_p, curs_v, curs_a = self.resample_kin(new_bin_size)
            if len(t_curs) > len(t_spike):
                curs_p = curs_p[:len(t_spike), :]
                curs_v = curs_v[:len(t_spike), :]
                curs_a = curs_a[:len(t_spike), :]
        
        if update == 1:
            self.time_frame = t_spike
            self.bin_width = new_bin_size
            self.spike_counts = spike_counts
            if self.has_EMG == 1:
                self.EMG = EMG
            if self.has_force == 1:
                self.force = force
            if flag_cursor == 1:
                if hasattr(self, 'curs_p'):
                    self.curs_p, self.curs_v, self.curs_a = curs_p, curs_v, curs_a
                elif hasattr(self, 'kin_p'):
                    self.kin_p, self.kin_v, self.kin_a = curs_p, curs_v, curs_a
    
    def bin_spikes(self, bin_size, mode = 'center'):
        print('The new bin size is %.4f s' % (bin_size)) 
        spike_counts = []         
        if mode == 'center':
            bins = np.arange(self.time_frame[0] - bin_size/2, 
                             self.time_frame[-1] + bin_size/2, bin_size)
        elif mode == 'left':
            bins = np.arange(self.time_frame[0], self.time_frame[-1], bin_size)
        bins = bins.reshape((len(bins),))
        for each in self.spikes:
            bb=each.reshape((len(each),))
            out, _ = np.histogram(bb, bins)
            spike_counts.append(out)
        bins = bins.reshape((len(bins),1))
        return bins[1:], np.asarray(spike_counts).T
              
    def resample_EMG(self, new_bin_size):
        """
        Downsampling the filtered EMG
        """
        if self.has_EMG == 0:
            print('There is no EMG in this file.')
            return 0
        else:
            if new_bin_size < self.bin_width:
                print('Cannot bin EMG using this bin size')
                return 0
            else:
                n = new_bin_size/self.bin_width
                L = int(np.floor(self.EMG.shape[0]/n))
                idx = [int(np.floor(i*n)) for i in range(1, L)]
                return self.time_frame[idx], self.EMG[idx, :]

    def resample_force(self, new_bin_size):
        if self.has_force == 0:
            print('There is no force in this file.')
            return 0
        else:
            if new_bin_size < self.bin_width:
                print('Cannot bin force using this bin size')
                return 0
            else:
                n = new_bin_size/self.bin_width
                L = int(np.floor(np.size(self.force, 0)/n))
                idx = [int(np.floor(i*n)) for i in range(1, L)]
                return self.time_frame[idx], self.force[idx, :]

    def resample_kin(self, new_bin_size):
        if hasattr(self, 'has_cursor'):
            flag = self.has_cursor
        elif hasattr(self, 'has_kin'):
            flag = self.has_kin
        
        if flag == 0:
            print('There is no kinematics in this file.')
            return 0
        else:
            if new_bin_size < self.bin_width:
                print('Cannot bin kinematics using this bin size')
                return 0
            else:
                n = new_bin_size/self.bin_width
                if hasattr(self, 'curs_p'):
                    L = int(np.floor(self.curs_p.shape[0]/n))
                    idx = [int(np.floor(i*n)) for i in range(1, L)]
                    return self.time_frame[idx], self.curs_p[idx, :], self.curs_v[idx, :], self.curs_a[idx, :]
                elif hasattr(self, 'kin_p'):
                    L = int(np.floor(self.kin_p.shape[0]/n))
                    idx = [int(np.floor(i*n)) for i in range(1, L)]
                    return self.time_frame[idx], self.kin_p[idx, :], self.kin_v[idx, :], self.kin_a[idx, :]

    def smooth_binned_spikes(self, bin_size, kernel_type, kernel_SD, sqrt = 0):
        binned_spikes = self.spike_counts.T.tolist()
        smoothed = []
        if sqrt == 1:
            for (i, each) in enumerate(binned_spikes):
                binned_spikes[i] = np.sqrt(each)
        kernel_hl = 3 * int( kernel_SD / bin_size )
        normalDistribution = stats.norm(0, kernel_SD)
        x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
        kernel = normalDistribution.pdf(x)
        if kernel_type == 'gaussian':
            pass
        elif kernel_type == 'half_gaussian':
            for i in range(0, int(kernel_hl)):
                kernel[i] = 0
        n_sample = np.size(binned_spikes[0])
        nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)] 
        for each in binned_spikes:
            temp1 = np.convolve(kernel,each)
            temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
            smoothed.append(temp2)
        #print('The spike counts have been smoothed.')
        self.spike_counts = np.asarray(smoothed).T

    def sort_trials_target_dir(self, data_type, target_dir_list, trial_type, time_params, EMG_channels = 'all'):
        start_event = time_params['start_event']
        ahead = time_params['time_before_start']
        end_event = time_params['end_event']
        end_time_offset = time_params['time_after_end']
        if data_type == 'spike_counts':
            data = self.get_trials_data_spike_counts(trial_type, start_event, ahead, end_event, end_time_offset)
        elif data_type == 'EMG':
            try:
                data = self.get_trials_data_EMG(trial_type, start_event, ahead, end_event, end_time_offset, EMG_channels)
            except Exception:
                print('No EMG in this file')
        elif data_type == 'raw_EMG':
            try:
                data = self.get_trials_data_raw_EMG(trial_type, start_event, ahead, end_event, end_time_offset, EMG_channels)
            except Exception:
                print('No raw EMG in this file')
        elif data_type == 'force':
            try:
                data = self.get_trials_data_force(trial_type, start_event, ahead, end_event, end_time_offset)
            except Exception:
                print('No force in this file')
        elif data_type == 'cursor':
            try:
                data = self.get_trials_data_kin(trial_type, start_event, ahead, end_event, end_time_offset)
            except Exception:
                print('No cursor trajectories in this file')
        elif data_type == 'time_frame':
            data = self.get_trials_data_time_frame(trial_type, start_event, ahead, end_event, end_time_offset)
        elif data_type == 'spikes':
            data = self.get_trials_data_spikes(trial_type, start_event, ahead, end_event, end_time_offset)
        
        trial_type_idx = np.where(self.trial_result == 'R')[0]
        trial_all_target, trial_all_curs_p, trial_all_curs_v, trial_all_curs_a = [], [], [], []
        for each in target_dir_list:
            target_dir_idx = np.where(self.trial_target_dir == each)[0]
            b = list(set(trial_type_idx).intersection(set(target_dir_idx)))
            c = sorted([np.where(trial_type_idx == each)[0][0] for each in b])
            if data_type == 'cursor':
                trial_all_curs_p.append([data[0][i] for i in c])
                trial_all_curs_v.append([data[1][i] for i in c])
                trial_all_curs_a.append([data[2][i] for i in c])
            else:
                trial_all_target.append([data[i] for i in c])
        if data_type == 'cursor':
            return trial_all_curs_p, trial_all_curs_v, trial_all_curs_a
        else:      
            return trial_all_target
        
    def sort_trials_target_center(self, x_or_y, data_type, target_center_list, trial_type, time_params, EMG_channels = 'all', gadget_number = -1):
        start_event = time_params['start_event']
        ahead = time_params['time_before_start']
        end_event = time_params['end_event']
        end_time_offset = time_params['time_after_end']
        if data_type == 'spike_counts':
            data = self.get_trials_data_spike_counts(trial_type, start_event, ahead, end_event, end_time_offset)
        elif data_type == 'EMG':
            try:
                data = self.get_trials_data_EMG(trial_type, start_event, ahead, end_event, end_time_offset, EMG_channels)
            except Exception:
                print('No EMG in this file')
        elif data_type == 'raw_EMG':
            try:
                data = self.get_trials_data_raw_EMG(trial_type, start_event, ahead, end_event, end_time_offset, EMG_channels)
            except Exception:
                print('No raw EMG in this file')
        elif data_type == 'force':
            try:
                data = self.get_trials_data_force(trial_type, start_event, ahead, end_event, end_time_offset)
            except Exception:
                print('No force in this file')
        elif data_type == 'cursor':
            try:
                data = self.get_trials_data_kin(trial_type, start_event, ahead, end_event, end_time_offset)
            except Exception:
                print('No cursor trajectories in this file')
        elif data_type == 'time_frame':
            data = self.get_trials_data_time_frame(trial_type, start_event, ahead, end_event, end_time_offset)
        elif data_type == 'spikes':
            data = self.get_trials_data_spikes(trial_type, start_event, ahead, end_event, end_time_offset)
        
        trial_type_idx = np.where(self.trial_result == 'R')[0]
        trial_all_target, trial_all_curs_p, trial_all_curs_v, trial_all_curs_a = [], [], [], []
        for each in target_center_list:
            if x_or_y == 'x':
                target_center_idx = np.where(self.trial_target_center_x == each)[0]
            elif x_or_y == 'y':
                target_center_idx = np.where(self.trial_target_center_y == each)[0]
            b = list(set(trial_type_idx) & set(target_center_idx))
            if gadget_number != -1:
                gadget_number_idx = np.where(self.trial_gadget_number == gadget_number)[0]
                b = list(set(trial_type_idx) & set(target_center_idx) & set(gadget_number_idx))
            c = sorted([np.where(trial_type_idx == each)[0][0] for each in b])
            if data_type == 'cursor':
                trial_all_curs_p.append([data[0][i] for i in c])
                trial_all_curs_v.append([data[1][i] for i in c])
                trial_all_curs_a.append([data[2][i] for i in c])
            else:
                trial_all_target.append([data[i] for i in c])
        if data_type == 'cursor':
            return trial_all_curs_p, trial_all_curs_v, trial_all_curs_a
        else:      
            return trial_all_target
        
    def get_electrode_idx(self, elec_num):
        """
        To get the idx of electrodes specified by elec_num
        dataset: xds structure
        elec_num: a list containing the number of bad channels
        """
        idx = []
        for each in elec_num:
            if 'elec'+str(each) in self.unit_names:
                temp = self.unit_names.index('elec'+str(each))
                idx.append(temp)
        return idx

    def del_electrode(self, elec_num):
        """
        To get rid of everything about the bad channels from my_cage_data
        """
        idx = self.get_electrode_idx(elec_num)
        for d in sorted(idx, reverse=True):
            del(self.unit_names[d])
            del(self.spikes[d])
        self.spike_counts = np.delete(self.spike_counts, idx, axis = 1)
   
###################################################################################################################################
###################################################################################################################################
# -------- Starting from August, 2020, the EMGs during some in-lab recording sessions are recorded using DSPW system too -------- #
# -------- The codes below are designed to deal with those sessions --------#
###################################################################################################################################
###################################################################################################################################

Pop_EMG_names_single = ['APB_1', 'Lum_1', 'PT_1', '1DI_1',
                        'FDP2_1', 'FCR1_1', 'FCU1_1', 'FCUR_1',
                        'FCUR_2', 'FCU1_2',	'FCR1_2', 'FDP2_2', 
                        '1DI_2', 'PT_2', 'Lum_2', 'APB_2',
                        'FPB_1', '3DI_1', 'SUP_1', 'ECU_1', 
                        'ECR_1', 'EDC1_1',	'BI_1', 'TRI_1', 
                        'TRI_2', 'BI_2', 'EDC1_2', 'ECR_2',
                        'ECU_2', 'SUP_2', '3DI_2', 'FPB_2']
"""
For the datasets collected on Pop between 2020-03 and 2020-09 using the DSPW system, channels 7 and 16 are noisy and should be taken out.
For the datasets collected on Pop after 2020-09 using the DSPW system, channels 7, 16, 3, and 12 are noisy and should be taken out.
For the datasets collected on all monkeys after 2018-12, channels 24, 25, and 26 should be taken out due to the short circuit of the adapter board.
"""

"""
In summary, for the data collected between 2020-09 and 2020-10 on Pop, the indices and names for the bad EMG channels are as below:
indices: [3, 7, 12, 16, 24, 25, 26]
names: ['1DI_1', 'FCUR_1', '1DI_2', 'FPB_1', 'TRI_2', 'BI_2', 'EDC1_2']
"""        
class lab_data_DSPW_EMG(lab_data):
    def __init__(self, base_path, file_name, rhd_file_name, bad_EMG = [], bin_size = 0.001, comb_filter = 0, art_reject = 1):
        if base_path[-1] != '/':
            base_path = base_path + '/'
        super(lab_data_DSPW_EMG, self).__init__(base_path, file_name) 
        
        if not path.exists( base_path + rhd_file_name ):
            raise Exception( 'Can''t find the file for DSPW EMG:' + file_name )
        self.parse_file_DSPW_EMG(base_path + rhd_file_name, 1, bad_EMG, 10, comb_filter, art_reject)
        self.bin_from_rhd(bin_size, mode = 'center')
        self.spike_counts, self.EMG = np.asarray(self.spike_counts).T, np.asarray(self.EMG).T
            
    def parse_file_DSPW_EMG(self, file_name, notch, bad_EMG, f_lp = 10, comb_filter = 0, art_reject = 1):
        EMG_names, raw_EMG, raw_EMG_time_frame = self.parse_rhd_file(file_name, notch, bad_EMG, comb_filter, art_reject)
        self.EMG = self.EMG_filtering(raw_EMG, f_lp)
        self.has_EMG = 1
        self.EMG_names, self.raw_EMG, self.raw_EMG_time_frame = EMG_names, np.asarray(raw_EMG).T, raw_EMG_time_frame
        
    def parse_rhd_file(self, filename, notch, bad_EMG, comb_filter, art_reject):
        rhd_data = read_data(filename)
        self.EMG_fs = rhd_data['frequency_parameters']['amplifier_sample_rate']
        EMG_single = rhd_data['amplifier_data']
        
        # -------- To get the name for each individual EMG channel from the rhd file -------- #
        EMG_names_single = []
        for each in rhd_data['amplifier_channels']:
            EMG_names_single.append(each['custom_channel_name'])
        
        # -------- Determine whether the labels for each EMG channel are needed to be replaced -------- #
        meta_info = self.get_meta()
        collect_date = meta_info['collect_date']
        slash_pos = get_char_pos(collect_date, '/')
        if slash_pos[1][0]-slash_pos[0][0] == 2:
            date_num = collect_date[:4]+'0'+collect_date[5]
        elif slash_pos[1][0]-slash_pos[0][0] == 3: 
            date_num = collect_date[:4]+collect_date[5:7]
        if (int(date_num)>202003)&(int(date_num)<202011):
            EMG_names_single = copy.deepcopy(Pop_EMG_names_single)
            print('Using a fixed EMG channel definition for this dataset')
        
        # -------- If the items in bad_EMG are numbers, these lines will find out the names -------- #
        if len(bad_EMG) > 0:
            if type(bad_EMG[0]) == int:
                bad_EMG_names = [EMG_names_single[n] for n in bad_EMG]
            elif type(bad_EMG[0]) == str:
                bad_EMG_names = bad_EMG
        else:
            bad_EMG_names = []
        
        # ---------- Delete paired bad channels -------- #
        bad_paired_channel, bad_EMG_post = delete_paired_bad_channel(EMG_names_single, bad_EMG_names)
        bad_paired_channel = sorted(bad_paired_channel, reverse = True)
        for each in bad_paired_channel:
            EMG_names_single.pop(each)
        EMG_single = np.delete(EMG_single, bad_paired_channel, axis = 0)
        
        # ---------- To get paired EMG channels for software diffrence ---------- #
        EMG_names, EMG_index1, EMG_index2 = get_paired_EMG_index(EMG_names_single)

        EMG_diff = []
        for i in range(len(EMG_index1)):
            EMG_diff.append(EMG_single[EMG_index1[i], :] - EMG_single[EMG_index2[i], :])
        
        # ---------- Based on the list in bad_EMG, substitute some channels with single end EMG ---------- #
        if bad_EMG:
            bad_idx, paired_idx = find_bad_EMG_index_from_list(EMG_names_single, bad_EMG_post)
            for (i,each) in enumerate(bad_EMG_post):
                target_idx = EMG_names.index(each[:-2])
                EMG_diff[target_idx] = EMG_single[paired_idx[i], :]
                print("For noisy channel %s, use only one single end channel." %(each[:-2]))
                lost_idx = np.where(EMG_diff[target_idx]<-6300)[0]
                if lost_idx.size > 0:
                    EMG_diff[target_idx][lost_idx] = EMG_diff[target_idx][lost_idx[0]-10]
        
        # ---------- Apply artifacts rejection on EMG_diff ----------- #
        """
        For all dataset, artifacts rejection is necessary, must be done
        """
        if art_reject == 1:
            EMG_diff = self.EMG_art_rej(EMG_diff)        
        # ---------- Apply notch filter on EMG_diff ---------- #
        if notch == 1:
           print('Applying notch filter.')
           bnotch, anotch =  signal.iirnotch(60, 30, self.EMG_fs)
           for (i, each) in enumerate(EMG_diff): 
               EMG_diff[i] = signal.filtfilt(bnotch, anotch, each)
        else:
            print('No notch filter is applied.')
        
        # ---------- Apply comb filter on EMG_diff ----------- #
        if comb_filter == 1:
            EMG_diff = self.apply_comb_filter(EMG_diff, self.EMG_fs)
            print('Applying comb filter.')
        
        EMG_diff = np.asarray(EMG_diff)
        
        sync_line0 = rhd_data['board_dig_in_data'][0]
        sync_line1 = rhd_data['board_dig_in_data'][1]
        d0 = np.where(sync_line0 == True)[0]
        d1 = np.where(sync_line1 == True)[0]
        ds = int(d0[0])
        de = int(d1[-1])
        #ds = int(d1[0] - int((d1[0]-d0[0])*0.2))
        #de = int(d1[-1] + int((d0[-1]-d1[-1])*0.2))
        rhd_timeframe = np.arange(de-ds+1)/self.EMG_fs
        return EMG_names, list(EMG_diff[:, ds:de]), rhd_timeframe
    
    def EMG_filtering(self, raw_EMG_data, f_Hz, art_reject = 1):
        fs = self.EMG_fs
        filtered_EMG = []    
        bhigh, ahigh = signal.butter(4,50/(fs/2), 'high')
        blow, alow = signal.butter(4,f_Hz/(fs/2), 'low')
        for each in raw_EMG_data:
            temp = signal.filtfilt(bhigh, ahigh, each)
            if art_reject == 1:
                temp = self.EMG_art_rej_single_channel(temp)
            f_abs_emg = signal.filtfilt(blow ,alow, np.abs(temp))
            filtered_EMG.append(f_abs_emg)   
        print('All EMG channels have been filtered.')
        return filtered_EMG

    def bin_spikes_with_rhd(self, bin_size, mode = 'center'):
        print('Binning spikes with %.4f s' % (bin_size))
        binned_spikes = []
        bin_start = self.time_frame[0]
        if mode == 'center':
            bins = np.arange(bin_start - bin_size/2, 
                             self.time_frame[-1] + bin_size/2, bin_size)
        elif mode == 'left':
            bins = np.arange(bin_start, self.time_frame[-1], bin_size)
        bins = bins.reshape((len(bins),))
        for each in self.spikes:
            each = each.reshape((len(each),))
            out, _ = np.histogram(each, bins)
            binned_spikes.append(out)
        return bins[1:], binned_spikes        
      
    def EMG_downsample_rhd(self, new_fs):
        if hasattr(self, 'EMG'):
            down_sampled = []
            n = self.EMG_fs/new_fs
            L = int(np.floor(np.size(self.EMG[0])/n)+1)
            for each in self.EMG:
                temp = np.asarray([each[int(np.floor(i*n))] for i in range(1, L)])
                down_sampled.append(temp)
            print('Filtered EMGs have been downsampled')
            return down_sampled
        else:
            print('Filter EMG first!')
            return 0
        
    def raw_force_downsample(self, new_fs):
        """
        The sampling frequency for raw forces from Cerebus is calculated from the timeframe.
        Before downsampling, these signals need to be filtered at 10 Hz.
        """
        if self.has_raw_force == 1:
            fs = 1/(self.raw_force_time_frame[10] - self.raw_force_time_frame[9])
            blow, alow = signal.butter(4, 10/(fs/2), 'low')
            filtered = []
            for i in range(2):
                filtered.append( signal.filtfilt(blow, alow, np.abs(self.raw_force[:, i])) )
            filtered = np.asarray(filtered).T
            down_sampled = []
            n = fs/new_fs
            L = int(np.floor(np.size(filtered, 0)/n)+1)
            for i in range( 1, L ):
                down_sampled.append(filtered[int(np.floor(i*n)), :])
            return np.asarray(down_sampled)
        else:
            print('There are now raw force data in this file')
            return 0 
        
    def bin_from_rhd(self, bin_size, mode = 'center'):
        print('Bin data from both Cerebus recorded nev file and DSPW recorded rhd file')
        self.time_frame, self.spike_counts = self.bin_spikes_with_rhd(bin_size, mode)
        if self.has_EMG == 1:
            self.EMG = self.EMG_downsample_rhd(1/bin_size)
            # try:
            #     self.force = self.raw_force_downsample(1/bin_size)
            # except:
            #     print('Did not bin force data because there is no force')
            if self.has_force == 1:
                truncated_len = min(len(self.EMG[0]), len(self.spike_counts[0]), len(self.force))
            else:
                truncated_len = min(len(self.EMG[0]), len(self.spike_counts[0]))
            for (i, each) in enumerate(self.spike_counts):
                self.spike_counts[i] = each[:truncated_len]
            for (i, each) in enumerate(self.EMG):
                self.EMG[i] = each[:truncated_len]
            self.time_frame = self.time_frame[:truncated_len]
            if self.has_force == 1:
                self.force = self.force[:truncated_len, :]
        print('Data have been binned.')

    def apply_comb_filter(self, input_signal, fs, f_list = [120, 180, 240, 300, 360], Q = 30):
        """
        Here input_signal is a list
        """
        output_signal = input_signal
        b, a = [], []
        for i in range(len(f_list)):
            b_temp, a_temp = signal.iirnotch(f_list[i], Q, fs)
            b.append(b_temp)
            a.append(a_temp)
        for i in range(len(input_signal)):
            for j in range(len(f_list)):
                output_signal[i] = signal.filtfilt(b[j], a[j], input_signal[i])
        return output_signal
        
    def EMG_art_rej(self, data_list, k = 8, L = 8):
        print('Rejecting high amplitude EMG artifacts.')
        data_list_post = []
        for data in data_list:
            c = np.where(abs(data)>k*np.std(data))[0]
            idx = []
            for each in c:
                idx.append(list(np.arange(each-L, each+L)))
            u_idx = sorted(set(idx[0]).union(*idx))
            u_idx = np.asarray(u_idx)
            over_idx = np.where(u_idx>len(data)-1)[0]
            u_idx = list(np.delete(u_idx, over_idx))
            subs = np.random.rand(len(u_idx))*np.std(data)
            data[u_idx] = subs
            data_list_post.append(data)
        return data_list_post     

    def EMG_art_rej_single_channel(self, data, k = 8, L = 8):
        #print('Rejecting high amplitude EMG artifacts on single channel.')
        c = np.where(abs(data)>k*np.std(data))[0]
        idx = []
        for each in c:
            idx.append(list(np.arange(each-L, each+L)))
        u_idx = sorted(set(idx[0]).union(*idx))
        u_idx = np.asarray(u_idx)
        over_idx = np.where(u_idx>len(data)-1)[0]
        u_idx = list(np.delete(u_idx, over_idx))
        subs = np.random.rand(len(u_idx))*np.std(data)
        data[u_idx] = subs
        return data                             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            