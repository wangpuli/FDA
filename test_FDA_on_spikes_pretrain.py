import os, sys, fnmatch
sys.path.append('./xds/xds_python/')

import numpy as np
import random
import torch
import argparse
from sklearn.model_selection import train_test_split

from preprocess.data_loader import load_monkey_spike, load_spike_generator, concatenate_spike_token
from utils.padding import spike_zero_padding
from condition.vanilla_iTransfomer import ConditionModel
from config.model_config import ModelConfig
from config.train_config import TrainConfig, TrainEmbedderConfig
from flow.models.SiT_models import SiT
from experiment.train_func import training_stage, train_dynamic_embedder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

## pre-train phase
if __name__ == '__main__':
    # parse arguments                    
    parser = argparse.ArgumentParser(description='train')
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
    src_train_ratio = 0.9
    training_step = 3500
    invert_flag = True if dataset_name in ['CO-M', 'RT-M'] else False

    ## source session pre-train
    # load src and tgt datas for max channels
    # max channel names
    max_unit_names = []
    for idx in range(len(mat_list)):
        src_data_date = mat_list[idx][id_len+1:id_len+date_len+1]   
        _, _, unit_names = load_monkey_spike(data_path, src_data_date, bin_size, smooth_size, start_time, NHP_id)
        
        max_unit_names = np.sort(list(set(max_unit_names)|set(unit_names)))
    
    # set the source session indexes (Day 0)
    src_session_idx = [0]
    # 0: position 1: velocity 2: accelerated speed
    cur_idx = 1

    src_day_spike, src_day_cursor_pos_xy = [], []
    for src_idx in src_session_idx:
        data_date = mat_list[src_idx][id_len+1:id_len+date_len+1]

        print("current srouce recording date: %s" % data_date)
        print("data preparing...")
        day_spike, day_cursor, unit_names = load_monkey_spike(data_path, data_date, bin_size, smooth_size, start_time, NHP_id)
        day_cursor_pos_xy = day_cursor[cur_idx]

        # zero-padding
        day_spike_tmp = spike_zero_padding(max_unit_names, unit_names, day_spike)

        src_day_spike.extend(day_spike_tmp)
        src_day_cursor_pos_xy.extend(day_cursor_pos_xy)

    setup_seed(3)
    train_day_spike, test_day_spike, train_day_cursor, test_day_cursor = train_test_split(src_day_spike, src_day_cursor_pos_xy, test_size=1.0-src_train_ratio, random_state=0)
    
    # generate dataloader
    fourier_flag = False
    batch_size = 256
    train_generator = load_spike_generator(day_spike=train_day_spike,
                                        day_cursor=train_day_cursor,
                                        window_size=window_size,
                                        batch_size=batch_size,
                                        is_shuffle=False,
                                        fourier_flag=fourier_flag)
    valid_day_spike_format, valid_day_cursor_format = concatenate_spike_token(test_day_spike, test_day_cursor, window_size, is_shuffle=False, fourier_flag=fourier_flag)
    valid_generator = (valid_day_spike_format, valid_day_cursor_format)

    device = torch.device("cuda:" + args.cuda_device) if torch.cuda.is_available() else torch.device('cpu')

    # conditional extractor
    context_size = window_size
    n_chan  = train_day_spike[0].shape[1] if not invert_flag else context_size
    seq_len = context_size if not invert_flag else train_day_spike[0].shape[1]
    configs = ModelConfig(
        seq_len=seq_len,
        enc_in=n_chan,
        training_step=training_step,
        e_layers=2,
        factor=1,
    )
    transformer_model = ConditionModel(configs)

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

    # cold start
    training_step = 600 if NHP_id == 'Chewie' else 200
    train_embdder_config = TrainEmbedderConfig(
        device=device,
        train_generator=train_generator,
        valid_generator=valid_generator,
        model=flow_model.dynamic_embedder.transformer_model,
        invert_flag=invert_flag,
        training_step=training_step,
        update_flag=True,
    )
    initial_model = train_dynamic_embedder(train_embdder_config)

    # training process settings
    train_config = TrainConfig(
        cfg_scale=1.0,
        device=device,
        train_generator=train_generator,
        valid_generator=valid_generator,
        model=flow_model,
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

    (best_valid_r2, valid_r2_epoch, valid_r2_curve), model, pre_train_results = training_stage(train_config)
    print('dataset name: %s, src date: %s, best valid r2: %.4f' % (dataset_name, data_date, best_valid_r2))

    # save pre-trained model
    ckpt_dir = f'./ckpt/{dataset_name}/pre_train'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({
        'model_state_dict': model.state_dict(),
        'pre_train_results': pre_train_results,
        'valid_r2_curve': valid_r2_curve,
        'valid_r2_epoch': valid_r2_epoch,
        'best_valid_r2': best_valid_r2,
    }, f'{ckpt_dir}/FDA_pretrain_src_{data_date}_best_valid_{best_valid_r2:.2f}.pth')