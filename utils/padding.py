import numpy as np

def spike_preprocessing(unit_names1, unit_names2, spike1, spike2):
    """
    unit_names1: unit names in the first dataset
    unit_names2: unit names in the second dataset
    spike1: a list, the spike counts for each trial in the first dataset
    spike2: a list, the spike counts for each trial in the second dataset
    """
    all_unit_names = np.sort(list(set(unit_names1)|set(unit_names2)))
    N_unit = len(all_unit_names)
    
    idx = [list(all_unit_names).index(e) for e in unit_names1]
    spike1_ = [np.zeros((s.shape[0], N_unit)) for s in spike1]
    for k in range(len(spike1)):
        spike1_[k][:, idx] = spike1[k]
        
    idx = [list(all_unit_names).index(e) for e in unit_names2]
    spike2_ = [np.zeros((s.shape[0], N_unit)) for s in spike2]
    for k in range(len(spike2)):
        spike2_[k][:, idx] = spike2[k]
    return spike1_, spike2_

def spike_zero_padding(max_unit_names, spike_unit_names, spike):
    N_unit = len(max_unit_names)
    
    idx = [list(max_unit_names).index(e) for e in spike_unit_names]
    spike_ = [np.zeros((s.shape[0], N_unit)) for s in spike]
    for k in range(len(spike)):
        spike_[k][:, idx] = spike[k]
    return spike_