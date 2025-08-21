import h5py
import numpy as np
import scipy.io as sio
from collections import defaultdict
import copy

def parse_h5py(path, file_name):
    """
    A function to parse h5py format .mat file (v7.3)
    """
    def arr_to_str(b):
        if isinstance(b, list) == 0:
           return ''.join([chr(int(each)) for each in b])
        else:
            b_ = []
            for each in b:
                b_.append(''.join([chr(c) for c in each]))
            return b_
    
    def parse_meta(raw_meta):
        meta = {}
        meta['ranBy'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['ranBy'])])
        meta['monkey'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['monkey'])])
        meta['array'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['array'])])
        meta['dateTime'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['dateTime'])])
        meta['processedTime'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['processedTime'])])
        meta['rawFileName'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['rawFileName'])])
        meta['task'] = ''.join([chr(int(each)) for each in np.asarray(raw_meta['task'])])
        meta['duration'] = np.asarray(raw_meta['duration'])[0][0]
        meta['lab'] = np.asarray(raw_meta['lab'])[0][0]
        meta['numTrials'] = np.asarray(raw_meta['numTrials'])[0][0]
        meta['numReward'] = np.asarray(raw_meta['numReward'])[0][0]
        meta['numAbort'] = np.asarray(raw_meta['numAbort'])[0][0]
        meta['numFail'] = np.asarray(raw_meta['numFail'])[0][0]
        meta['dataWindow'] = np.asarray(raw_meta['dataWindow'])
        return meta
        
    parsed = {}
    if path[-1] != '/':
        path = path + '/'
    read_data = h5py.File(path + file_name, 'r')
    xds = read_data['xds']
    
    parsed['has_EMG'] = np.asarray(xds['has_EMG'])[0][0]
    parsed['has_force'] = np.asarray(xds['has_force'])[0][0]
    if 'has_kin' in xds.keys():
        parsed['has_kin'] = np.asarray(xds['has_kin'])[0][0]
        parsed['has_cursor'] = np.asarray(xds['has_kin'])[0][0]
    if 'has_cursor' in xds.keys():
        parsed['has_cursor'] = np.asarray(xds['has_cursor'])[0][0]
    parsed['sorted'] = np.asarray(xds['sorted'])[0][0]
    parsed['bin_width'] = np.asarray(xds['bin_width'])[0][0]
    try:
        parsed['has_raw_EMG'] = np.asarray(xds['has_raw_EMG'])[0][0]
    except Exception:
        parsed['has_raw_EMG'] = 0
    try:
        parsed['has_raw_force'] = np.asarray(xds['has_raw_force'])[0][0]
    except Exception:
        parsed['has_raw_force'] = 0
    
    parsed['spikes'] = [read_data[xds['spikes'][i][0]][()].T for i in range(len(xds['spikes']))]
    parsed['spike_counts'] = np.asarray(xds['spike_counts'], dtype = np.uint8).T
    temp = [read_data[xds['unit_names'][i][0]][()] for i in range( len(xds['unit_names']) )]
    parsed['unit_names'] = [arr_to_str(each) for each in temp]
    parsed['time_frame'] = np.asarray(xds['time_frame']).reshape((-1, ))
    
    if 'spike_waveforms' in xds.keys():
        parsed['spike_waveforms'] = [read_data[xds['spike_waveforms'][i][0]][()].T for i in range(len(xds['spike_waveforms']))]
    if 'EMG' in xds.keys():
        parsed['EMG'] = np.asarray(xds['EMG']).T
        temp = [read_data[xds['EMG_names'][i][0]][()] for i in range( len(xds['EMG_names']) )]
        parsed['EMG_names'] = [arr_to_str(each) for each in temp]
    if 'raw_EMG' in xds.keys():
        parsed['raw_EMG'] = np.asarray(xds['raw_EMG']).T
        parsed['raw_EMG_time_frame'] = np.asarray(xds['raw_EMG_time_frame']).reshape((-1, ))
    if 'force' in xds.keys():
        parsed['force'] = np.asarray(xds['force']).T
    if 'raw_force' in xds.keys():
        parsed['raw_force'] = np.asarray(xds['raw_force']).T
        parsed['raw_force_time_frame'] = np.asarray(xds['raw_force_time_frame']).reshape((-1, ))
    if 'curs_p' in xds.keys():
        parsed['curs_p'] = np.asarray(xds['curs_p']).T
        parsed['curs_v'] = np.asarray(xds['curs_v']).T
        parsed['curs_a'] = np.asarray(xds['curs_a']).T
    elif 'kin_p' in xds.keys():
        parsed['curs_p'] = np.asarray(xds['kin_p']).T
        parsed['curs_v'] = np.asarray(xds['kin_v']).T
        parsed['curs_a'] = np.asarray(xds['kin_a']).T
    # -------- in some files this flag is not right, so we need to fix it -------- #
    if 'has_cursor' in parsed.keys():
        if parsed['has_cursor'] == 0:
            if 'curs_p' in parsed.keys():
                if len(parsed['curs_p'])>0:
                    parsed['has_cursor'] = 1
    
    # -------- handling trial timing related variables -------- #
    parsed['trial_start_time'] = np.asarray(xds['trial_start_time']).reshape((-1, ))
    parsed['trial_end_time'] = np.asarray(xds['trial_end_time']).reshape((-1, ))
    # -------- gocue_time may have different format from different tasks -------- #
    if xds['trial_gocue_time'].shape[0] == 1:
        parsed['trial_gocue_time'] = np.asarray(xds['trial_gocue_time']).reshape((-1, ))
    elif xds['trial_gocue_time'].shape[0] > 1:
        parsed['trial_gocue_time'] = np.asarray(xds['trial_gocue_time']).T
    else:
        print('something wrong with the gocue time')
    # -------- handling target related variables -------- #
    parsed['trial_target_dir'] = np.asarray(xds['trial_target_dir']).reshape((-1, ))
    parsed['trial_target_corners'] = np.asarray(xds['trial_target_corners']).T
    parsed['trial_result'] = np.asarray([chr(each) for each in np.asarray(xds['trial_result']).reshape((-1, ))])
    
    temp = [read_data[xds['trial_info_table_header'][0][i]][()] for i in range( xds['trial_info_table_header'].shape[1] )]
    parsed['trial_info_table_header'] = [arr_to_str(each) for each in temp]
    
    r, c = xds['trial_info_table'].shape
    trial_info_table = []
    for i in range(r):
        temp = []
        for j in range(c):
            temp.append(read_data[xds['trial_info_table'][i][j]][()])
        trial_info_table.append(temp)
    parsed['trial_info_table'] = trial_info_table
    
    parsed['meta'] = parse_meta(xds['meta'])
    
    read_data.close()
    return parsed

def parse_scipy(path, file_name):
    parsed = {}
    if path[-1] != '/':
        path = path + '/'
    read_data = sio.loadmat(path + file_name)
    xds = read_data['xds']
    
    # -------- meta information -------- #
    parsed['meta'] = {}
    parsed['meta']['monkey'] = xds['meta'][0][0]['monkey'][0][0][0]
    parsed['meta']['task'] = xds['meta'][0][0]['task'][0][0][0]
    parsed['meta']['duration'] = xds['meta'][0][0]['duration'][0][0][0][0]
    parsed['meta']['dateTime'] = xds['meta'][0][0]['dateTime'][0][0][0]
    parsed['meta']['rawFileName'] = xds['meta'][0][0]['rawFileName'][0][0][0]
    parsed['meta']['array'] = xds['meta'][0][0]['array'][0][0][0]
    parsed['meta']['processedTime'] = xds['meta'][0][0]['processedTime'][0][0][0]
    parsed['meta']['ranBy'] = xds['meta'][0][0]['ranBy'][0][0][0]
    parsed['meta']['lab'] = xds['meta'][0][0]['lab'][0][0][0][0]
    parsed['meta']['numTrials'] = xds['meta'][0][0]['numTrials'][0][0][0][0]
    parsed['meta']['numReward'] = xds['meta'][0][0]['numReward'][0][0][0][0]
    parsed['meta']['numAbort'] = xds['meta'][0][0]['numAbort'][0][0][0][0]
    parsed['meta']['numFail'] = xds['meta'][0][0]['numFail'][0][0][0][0]
    parsed['meta']['dataWindow'] = xds['meta'][0][0]['dataWindow'][0][0][0]
    
    # -------- flag variables -------- #
    parsed['has_EMG'] = xds['has_EMG'][0][0][0][0]
    parsed['has_kin'] = xds['has_kin'][0][0][0][0]
    parsed['has_cursor'] = parsed['has_kin']
    parsed['has_force'] = xds['has_force'][0][0][0][0]
    parsed['sorted'] = xds['sorted'][0][0][0][0]
    parsed['bin_width'] = xds['bin_width'][0][0][0][0]
    try:
        parsed['has_raw_force'] = xds['has_raw_force'][0][0][0][0]
    except Exception:
        parsed['has_raw_force'] = 0
    try:
        parsed['has_raw_EMG'] = xds['has_raw_EMG'][0][0][0][0]
    except Exception:
        parsed['has_raw_EMG'] = 0
    
    # -------- data -------- #
    parsed['time_frame'] = xds['time_frame'][0][0].reshape((-1, ))
    parsed['spike_counts'] = xds['spike_counts'][0][0]
    parsed['spikes'] = xds['spikes'][0][0][0].tolist()
    parsed['unit_names'] = [each.tolist()[0] for each in xds['unit_names'][0][0][0]]
    try:
        parsed['spike_waveforms'] = xds['spike_waveforms'][0][0][0].tolist()
    except Exception:
        pass
    try:
        parsed['EMG'] = xds['EMG'][0][0]
        parsed['EMG_names'] = [each.tolist()[0] for each in xds['EMG_names'][0][0][0]]
    except Exception:
        pass
    try:
        parsed['force'] = xds['force'][0][0]
    except Exception:
        pass
    try:
        parsed['curs_p'] = xds['curs_p'][0][0]
        parsed['curs_v'] = xds['curs_v'][0][0]
        parsed['curs_a'] = xds['curs_a'][0][0]
    except Exception:
        try:
           parsed['curs_p'] = xds['kin_p'][0][0]
           parsed['curs_v'] = xds['kin_v'][0][0]
           parsed['curs_a'] = xds['kin_a'][0][0]
        except Exception:
            pass
    # -------- in some files this flag is not right, so we need to fix it -------- #
    if parsed['has_cursor'] == 0:
        if 'curs_p' in parsed.keys():
            if len(parsed['curs_p'])>0:
                parsed['has_cursor'] = 1
    
    try:
        parsed['raw_EMG'] = xds['raw_EMG'][0][0]
        parsed['raw_EMG_time_frame'] = xds['raw_EMG_time_frame'][0][0]
    except Exception:
        pass
    try:
        parsed['raw_force'] = xds['raw_force'][0][0]
        parsed['raw_force_time_frame'] = xds['raw_force_time_frame'][0][0]
    except Exception:
        pass
    # -------- trial information -------- #
    parsed['trial_start_time'] = xds['trial_start_time'][0][0]
    parsed['trial_end_time'] = xds['trial_end_time'][0][0]
    parsed['trial_gocue_time'] = xds['trial_gocue_time'][0][0]
    parsed['trial_target_corners'] = xds['trial_target_corners'][0][0]
    parsed['trial_target_dir'] = xds['trial_target_dir'][0][0]
    parsed['trial_result'] = xds['trial_result'][0][0]
    parsed['trial_info_table'] = xds['trial_info_table'][0][0].T.tolist()
    parsed['trial_info_table_header'] = [str(each.tolist()[0][0]) for each in xds['trial_info_table_header'][0][0]]
        
    return parsed

def get_char_pos(string, char):
  chPos=[]
  try:
    chPos=list(((pos,char) for pos,val in enumerate(string) if(val == char)))
  except:
    pass
  return chPos

def get_paired_EMG_index(EMG_names_single):
    EMG_names = []
    EMG_index1 = []
    EMG_index2 = []
    for i in range(len(EMG_names_single)):
        temp_str = EMG_names_single[i][:-2]
        if temp_str in EMG_names:
            continue
        else:
            for j in range(i+1, len(EMG_names_single)):
                temp_str2 = EMG_names_single[j]
                if temp_str2.find(temp_str) != -1:
                    if (temp_str2[:-2] in EMG_names) == True:
                        EMG_names.append(''.join( (temp_str2[:-2], '-3') ))
                    else:
                        EMG_names.append(temp_str)
                    EMG_index1.append(EMG_names_single.index(EMG_names_single[i]))
                    EMG_index2.append(EMG_names_single.index(EMG_names_single[j]))
    return EMG_names, EMG_index1, EMG_index2

def find_bad_EMG_index_from_list(EMG_names_single, bad_EMG):
    bad_index = []
    paired_index = []
    for each in bad_EMG:
        temp = list(each)
        if each[-1] == '1':
           temp[-1] = '2'
        elif each[-1] == '2':
            temp[-1] = '1'
        elif each[-1] == '3':
            temp[-1] = '1'
        paired_name = ''.join(temp)
        # -------- Make sure the paired EMG channel can be found ---------- #
        if paired_name in EMG_names_single:
            bad_index.append(EMG_names_single.index(each))
            paired_index.append(EMG_names_single.index(paired_name))
    return bad_index, paired_index

def delete_paired_bad_channel(EMG_names_single, bad_EMG):
    """
    If both of the two single end channels are noise, then we need to get rid of both
    This function will find out the indices of them. Deleting will be done outside of this function
    """
    def list_duplicates(seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return ((key,locs) for key,locs in tally.items() if len(locs)>1)

    temp = []
    for each in bad_EMG:
        temp.append(each[:-1])
    bad_paired_channel = []
    names = []
    for dup in sorted(list_duplicates(temp)):
        print( 'The paired channels of %s will be deleted'%(dup[0]) )
        name1, name2 = bad_EMG[dup[1][0]], bad_EMG[dup[1][1]]
        names.append(name1)
        names.append(name2)
        bad_paired_channel.append(EMG_names_single.index(name1))
        bad_paired_channel.append(EMG_names_single.index(name2))
    bad_EMG_post = copy.deepcopy(bad_EMG)
    for each in names:
        bad_EMG_post.remove(each)
    print('The numbers of these bad channels are %s' % (bad_paired_channel))
    return bad_paired_channel, bad_EMG_post

def find_force_onset(force_list, ch, thr = 0.4):
    """
    This function is designed to find out the force onset time for each trial
    force_list: a list of force signals, each element corresponds to a trial
    ch: the number used to specify which force channel to use, typically 0 or 1, for Fx or Fy
    thr: the threshold to determine whether the force is on
    """
    onset_num = []
    for each in force_list:
        f = np.sqrt(each[:, 0]**2 + each[:, 1]**2)
        df = np.diff(f)
        temp = np.where(df >= thr*np.max(df))[0]              
        if len(temp) == 0:
            onset_num.append(0)
        else:
            onset_num.append(temp[0])
    return onset_num                 

def find_movement_onset(var_list, ch, thr):
    """
    The design of this function is the same as the function above, but not just for force signals
    the var_list is supposed to be a list containing movement trajectories, or velocities
    """
    onset_num = []
    for each in var_list:
        f = np.sqrt(each[:, 0]**2 + each[:, 1]**2)
        df = np.diff(f)
        temp = np.where(df >= thr*np.max(df))[0]              
        if len(temp) == 0:
            onset_num.append(0)
        else:
            onset_num.append(temp[0])
    return onset_num                 
    
def find_target_dir(trial_curs_p, target_list = [-135, -90, -45, 0, 45, 90, 135, 180]):
    """
    This function is used to find out the directions of trials, because sometimes they are wrong in origninal recordings
    curs_p: a list containing the cursor trajectories for each trial
    """    
    dirs = []
    for each in trial_curs_p:
        s = each[-1, :] - each[0, :]
        x, y = s[0], s[1]
        if (x>0)&(y==0):
            dirs.append(0)
        elif (x<0)&(y==0):
            dirs.append(180)
        elif (x==0)&(y>0):
            dirs.append(90)
        elif (x==0)&(y<0):
            dirs.append(-90)
        elif (x>0)&(y!=0):
            dirs.append(np.arctan(y/x)*180/np.pi)  
        elif (x<0)&(y!=0):
            dirs.append(np.arctan(y/x)*180/np.pi+y/abs(y)*180)
    dirs_ = []
    for each in dirs:
        if each<=-157.5:
            dirs_.append(180)
        else:
            idx = np.argmin(abs(target_list - each))
            dirs_.append(target_list[idx])
    return np.array(dirs_)
    
    
    
    
    
    
    
    
    
    
    
    



