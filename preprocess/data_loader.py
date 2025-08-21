from xds import lab_data
from sklearn.utils import shuffle
from scipy.fft import fft

import numpy as np

def load_monkey_spike(data_path, data_date, bin_size, smooth_size, start_time='start_time', NHP_id='Jango'):
    data_file = NHP_id + '_' + data_date + '_001.mat'
    day_data = lab_data(data_path, data_file)
    day_data.update_bin_data(bin_size) # Bin the spikes with the specified bin_size
    day_data.smooth_binned_spikes(bin_size, 'gaussian', smooth_size) # Smooth the binned spike counts
    day_unit_names = day_data.unit_names
    #-------- Extract smoothed spike counts in trials without temporal alignment --------#
    day_spike = day_data.get_trials_data_spike_counts('R', start_time, 0.0, 'end_time', 0)
    #-------- Extract cursor trajectory in trials --------#
    day_cursor = day_data.get_trials_data_cursor('R', start_time, 0.0, 'end_time', 0)
    return day_spike, day_cursor, day_unit_names

def tokenize_spike(day_spike, day_cursor, window_size):
    trial_num = len(day_spike)
    
    time_segment_list, pos_label_list = [], []
    for tr_idx in range(trial_num):
        trial_spike, trial_cursor = day_spike[tr_idx], day_cursor[tr_idx]
        time_sample = []
        label_sample = trial_cursor[(window_size -1):(trial_spike.shape[0]), :]
        label_sample = np.array(label_sample).astype(np.float32)
        for time_idx in range(window_size -1, trial_spike.shape[0]):
            a = trial_spike[(time_idx - window_size + 1):(time_idx+1), :]
            time_sample.append(a)
            
        time_sample = np.array(time_sample).astype(np.float32)
        time_segment_list.append(time_sample)
        pos_label_list.append(label_sample)

    return time_segment_list, pos_label_list

def concatenate_spike_token(day_spike, day_cursor, window_size, is_shuffle=False, fourier_flag=False):
    time_segment_list, pos_label_list = tokenize_spike(day_spike, day_cursor, window_size)

    trial_num = len(time_segment_list)
    for i in range(0, trial_num):
        # Fourier transform
        if fourier_flag:
            trial_neuron_freq = fft(time_segment_list[i], axis=1)
            # normalization
            time_segment_list[i] = trial_neuron_freq.real

        if i == 0:
            time_segment_format, pos_label_format = time_segment_list[0], pos_label_list[0]
        else:
            time_segment_format = np.concatenate((time_segment_format, time_segment_list[i]), axis=0)
            pos_label_format = np.concatenate((pos_label_format, pos_label_list[i]), axis=0) 

    return time_segment_format, pos_label_format

def load_spike_generator(day_spike, day_cursor, window_size, batch_size, is_shuffle=False, fourier_flag=False):
    time_segment_format, pos_label_format = concatenate_spike_token(day_spike, day_cursor, window_size, is_shuffle, fourier_flag)

    batch_cnt = 0
    while True:
        if batch_cnt*batch_size >= time_segment_format.shape[0]:
            batch_cnt = 0

        start_index = batch_cnt*batch_size
        end_index = min(start_index + batch_size, time_segment_format.shape[0])
        batch_spike = time_segment_format[start_index:end_index]
            
        batch_label = pos_label_format[start_index:end_index]
        batch_num = np.array(end_index - start_index)
        batch_cnt += 1

        # return a iterator
        yield batch_spike, batch_label, batch_num

def load_sample_generator(day_spike, day_cursor, window_size, batch_size, is_shuffle=False, fourier_flag=False):
    time_segment_format, pos_label_format = concatenate_spike_token(day_spike, day_cursor, window_size, is_shuffle, fourier_flag)

    batch_cnt = 0
    while batch_cnt*batch_size < time_segment_format.shape[0]:
        start_index = batch_cnt*batch_size
        end_index = min(start_index + batch_size, time_segment_format.shape[0])
        batch_spike = time_segment_format[start_index:end_index]
            
        batch_label = pos_label_format[start_index:end_index]
        batch_num = np.array(end_index - start_index)
        batch_cnt += 1

        # return a iterator
        yield batch_spike, batch_label, batch_num

def monkey_spike_transform(day_spike, day_cursor, window_size):
    trial_num = len(day_spike)
    
    time_segment_list, pos_label_list = [], []
    for tr_idx in range(trial_num):
        trial_spike, trial_cursor = day_spike[tr_idx], day_cursor[tr_idx]
        time_sample = []
        label_sample = trial_cursor[(window_size -1):(trial_spike.shape[0]), :]
        label_sample = np.array(label_sample).astype(np.float32)
        for time_idx in range(window_size -1, trial_spike.shape[0]):
            a = trial_spike[(time_idx - window_size + 1):(time_idx+1), :]
            time_sample.append(a)
            
        time_sample = np.array(time_sample).astype(np.float32)
        time_segment_list.append(time_sample)
        pos_label_list.append(label_sample)

    return time_segment_list, pos_label_list

def monkey_cursor_transform(day_cursor, day_target_idx, window_size, time_delay=5):
    trial_num = len(day_cursor)
    
    cursor_segment_list, target_idx_segment_list = [], []
    for tr_idx in range(trial_num):
        trial_cursor, trial_dir_idx = day_cursor[tr_idx], day_target_idx[tr_idx]
        cursor_sample = []
        target_idx_sample = []
        # target_idx_sample = np.zeros(shape=(trial_cursor.shape[0]-window_size+1, dir_size))
        # target_idx_sample[:, trial_dir_idx] = np.ones(target_idx_sample.shape[0])
        # time_delay = 4
        for time_idx in range(window_size-1, trial_cursor.shape[0]-time_delay):
            a = trial_cursor[(time_idx - window_size + 1):(time_idx+1), :]
            cursor_sample.append(a)

            # target_idx_sample.append(trial_cursor[time_idx+1, :])
            target_idx_sample.append(trial_dir_idx[time_idx+time_delay, :])
            
        cursor_sample = np.array(cursor_sample).astype(np.float32)
        cursor_segment_list.append(cursor_sample)

        target_idx_sample = np.array(target_idx_sample).astype(np.float32)
        target_idx_segment_list.append(target_idx_sample)

def format_monkey_spike_from_trial(time_segment_list, pos_label_list):
    trial_num = len(time_segment_list)
    time_segment_format, pos_label_format = time_segment_list[0], pos_label_list[0]
    for i in range(1, trial_num):
        time_segment_format = np.concatenate((time_segment_format, time_segment_list[i]), axis=0)
        pos_label_format = np.concatenate((pos_label_format, pos_label_list[i]), axis=0)

    return time_segment_format, pos_label_format 

def get_all_input_spike(day_spike, day_cursor, window_size, is_shuffle=False, is_pos=False):
    if not is_pos:
        time_segment_list, pos_label_list = monkey_spike_transform(day_spike, day_cursor, window_size)
    else:
        time_segment_list, pos_label_list = monkey_cursor_transform(day_spike, day_cursor, window_size)
    time_segment_format, pos_label_format = format_monkey_spike_from_trial(time_segment_list, pos_label_list)
    
    if is_shuffle:
        time_segment_format, pos_label_format = shuffle(time_segment_format, pos_label_format)

    return time_segment_format, pos_label_format