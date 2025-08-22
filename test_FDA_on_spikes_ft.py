import os, sys, fnmatch
sys.path.append('./xds/xds_python/')

import argparse
import glob
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess.data_loader import load_monkey_spike, load_spike_generator, concatenate_spike_token
from utils.padding import spike_zero_padding
from condition.vanilla_iTransfomer import ConditionModel
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from config.align_config import AlignConfig
from flow.models.SiT_models import SiT
from experiment.ft_func import fine_tuning_stage
from align.mmd import MMD_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

## fine-tune
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='finetune')
    parser.add_argument('-cuda_device', type=str, default='0', help='which gpu to use ')
    
    args = parser.parse_args()

    # load dataset
    # CO-M: Mihili_CO_2014, CO-C: Chewie_CO_2016, RT-M: Mihili_RT_2013_2014
    data_path_dict = {
        'CO-M': './datasets/Mihili_CO_2014/',
        'CO-C': './datasets/Chewie_CO_2016/',
        'RT-M': './datasets/Mihili_RT_2013_2014/',
    }
    
    # set dataset name
    dataset_name = 'RT-M'
    data_path = data_path_dict[dataset_name]
    NHP_id = 'Mihili' if dataset_name in ['CO-M', 'RT-M'] else 'Chewie'
    save_NHP_id = NHP_id if dataset_name in ['CO-M', 'CO-C'] else 'Mihili_RT'

    id_len, date_len = len(NHP_id), 8
    mat_list = np.sort(fnmatch.filter(os.listdir(data_path), "*.mat")) # We sorted the files by name
    
    # pre-process
    bin_size, smooth_size = 0.05, 0.1
    start_time = 'gocue_time'

    #  hyper-parameters
    window_size = 6 if NHP_id == 'Chewie' else 5 # the length of context windows
    hidden_size = 64 if NHP_id == 'Chewie' else 32 # the dimension of latent space
    invert_flag = True if dataset_name in ['CO-M', 'RT-M'] else False

    # max channel names
    max_unit_names = []
    for idx in range(len(mat_list)):
        src_data_date = mat_list[idx][id_len+1:id_len+date_len+1]   
        _, _, unit_names = load_monkey_spike(data_path, src_data_date, bin_size, smooth_size, start_time, NHP_id)
        
        max_unit_names = np.sort(list(set(max_unit_names)|set(unit_names)))
    
    # align config setting
    aligner_method = 'mmd'
    kernel_mul_dict={
        'Chewie':2.0,
        'Mihili': 1.0,
        'Mihili_RT': 2.0,
    }
    mmd_loss = None
    if aligner_method == 'mmd':
        kernel_mul, kernel_num = kernel_mul_dict[save_NHP_id], 5
        mmd_loss = MMD_loss(kernel_mul, kernel_num)

    # ft settings
    setup_seed(3)
    fine_tuning_step = 25
    tgt_train_ratio = 0.02
    device = torch.device("cuda:" + args.cuda_device) if torch.cuda.is_available() else torch.device('cpu')

    # load src and tgt datas
    # src session indexes
    src_session_idx = [0]
    # 0: position 1: velocity 2: accelerated speed
    cur_idx = 1

    src_day_spike, src_day_cursor_pos_xy = [], []
    for src_idx in src_session_idx:
        src_data_date = mat_list[src_idx][id_len+1:id_len+date_len+1]

        print("current srouce recording date: %s" % src_data_date)
        print("data preparing...")
        day_spike, day_cursor, unit_names = load_monkey_spike(data_path, src_data_date, bin_size, smooth_size, start_time, NHP_id)
        day_cursor_pos_xy = day_cursor[cur_idx]
    
        # zero-padding
        day_spike_tmp = spike_zero_padding(max_unit_names, unit_names, day_spike)

        src_day_spike.extend(day_spike_tmp)
        src_day_cursor_pos_xy.extend(day_cursor_pos_xy)
    
    # src dataloader
    fourier_flag = False
    batch_size = 256
    shuffle_flag = False
    src_train_generator = load_spike_generator(
        day_spike=src_day_spike,
        day_cursor=src_day_cursor_pos_xy,
        window_size=window_size,
        batch_size=batch_size,
        is_shuffle=shuffle_flag,
        fourier_flag=fourier_flag,
    )

    # conditional extractor
    context_size = window_size
    n_chan  = src_day_spike[0].shape[1] if not invert_flag else context_size
    seq_len = context_size if not invert_flag else src_day_spike[0].shape[1]
    configs = ModelConfig(
        seq_len=seq_len,
        enc_in=n_chan,
        e_layers=2,
        factor=1,
    )
    transformer_model = ConditionModel(configs)

    # training process settings
    pre_train_config = TrainConfig(
        cfg_scale=1.0,
        device=device,
        path_type="Linear",
        loss_weight=None,
        prediction="velocity",
        training_step=3500,
        weight_decay=1e-5,
        num_sampling_steps=2,
        sampling_method="euler",
        sample_every=20,
        ema_decay=0.99,
    )

    # flow-based model settings
    flow_model = SiT(
        in_channels=n_chan if not invert_flag else seq_len,
        window_size=context_size,
        hidden_size=hidden_size,
        out_dim=2,
        depth=4,
        mlp_ratio=2.0,
        model_config=configs,
        invert_flag=invert_flag,
    )

    # load tgt session
    for tgt_idx in range(len(mat_list)):
        if tgt_idx in src_session_idx:
            continue
        
        tgt_data_date = mat_list[tgt_idx][id_len+1:id_len+date_len+1]
        tgt_day_spike, tgt_day_cursor, tgt_unit_names = load_monkey_spike(data_path, tgt_data_date, bin_size, smooth_size, start_time, NHP_id)
        max_unit_names = np.sort(list(set(max_unit_names)|set(tgt_unit_names)))

        # zero-padding
        print("current target recording date: %s" % tgt_data_date)
        tgt_day_cursor_pos_xy = tgt_day_cursor[cur_idx]
        tgt_day_spike = spike_zero_padding(max_unit_names, tgt_unit_names, tgt_day_spike)

        # load pre-trained model
        pre_weight_pth = f'./ckpt/{dataset_name}/pre_train/FDA_pretrain_src_{src_data_date}_best_valid_*.pth'
        # search pth
        matching_files = glob.glob(pre_weight_pth)
        if matching_files:
            pre_weight_pth = matching_files[0]
            pre_model= torch.load(pre_weight_pth)
            flow_model.load_state_dict(pre_model['model_state_dict'])
        else:
            raise ValueError(f"No pre-trained model found for {pre_weight_pth}")

        # tgt dataloader
        tgt_day_spike_ft, tgt_day_spike_test, tgt_day_cursor_ft, tgt_day_cursor_test = train_test_split(tgt_day_spike, tgt_day_cursor_pos_xy, test_size=0.8, random_state=3)

        tgt_train_num = int(len(tgt_day_spike)*tgt_train_ratio)
        tgt_day_spike_train, tgt_day_cursor_train = tgt_day_spike_ft[:tgt_train_num], tgt_day_cursor_ft[:tgt_train_num]

        tgt_day_spike_valid, tgt_day_cursor_valid = tgt_day_spike_ft[tgt_train_num:], tgt_day_cursor_ft[tgt_train_num:]

        tgt_train_generator = load_spike_generator(
            day_spike=tgt_day_spike_train,
            day_cursor=tgt_day_cursor_train,
            window_size=window_size,
            batch_size=batch_size,
            is_shuffle=shuffle_flag,
            fourier_flag=fourier_flag,
        )

        # valid & test
        tgt_day_spike_valid_format, tgt_cursor_valid_format = concatenate_spike_token(tgt_day_spike_valid, tgt_day_cursor_valid, window_size, is_shuffle=shuffle_flag, fourier_flag=fourier_flag)
        tgt_valid_generator = (tgt_day_spike_valid_format, tgt_cursor_valid_format)

        tgt_day_spike_test_format, tgt_cursor_test_format = concatenate_spike_token(tgt_day_spike_test, tgt_day_cursor_test, window_size, is_shuffle=shuffle_flag, fourier_flag=fourier_flag)
        tgt_test_generator = (tgt_day_spike_test_format, tgt_cursor_test_format)

        # align config
        learning_rate = 1e-4 if aligner_method == 'likelihood' else 5e-4
        align_config = AlignConfig(
            cfg_scale=1.0,
            num_sampling_steps=2,
            aligner_method=aligner_method,
            mmd_loss=mmd_loss,

            device=device,
            src_train_generator=src_train_generator,
            train_generator=tgt_train_generator,
            valid_generator=tgt_valid_generator,
            test_generator=tgt_test_generator,
            model=flow_model,
            pre_train_config=pre_train_config,

            tgt_train_ratio=tgt_train_ratio,
            fine_tuning_step=fine_tuning_step,
            learning_rate=learning_rate,
            weight_decay=1e-5,
            sample_every=1,
            sampling_method='euler',
        )
        (best_valid_r2, r2_score_valid_final, r2_score_test), best_valid_model = fine_tuning_stage(align_config)

        # save ft model
        ckpt_dir = f'./ckpt/{dataset_name}/ft'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save({
            'model_state_dict': best_valid_model.state_dict(),
            'best_valid_r2': best_valid_r2,
            'r2_score_valid_final': r2_score_valid_final,
            'r2_score_test': r2_score_test,
        }, f'{ckpt_dir}/FDA_ft_tgt_{tgt_data_date}_best_test_{r2_score_test:.2f}.pth')