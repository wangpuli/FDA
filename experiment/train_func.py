import torch
import time

from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy
from sklearn.metrics import r2_score
from flow.transport.transport import create_transport, Sampler

#################################################################################
#                                  Training Loop                                #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

# training in different epoches
def training_stage(args):
    # classifier-free guidance
    use_cfg = args.cfg_scale > 1.0

    device = args.device
    assert args.train_generator is not None
    train_generator = args.train_generator
    assert args.valid_generator is not None
    valid_generator = args.valid_generator
    assert args.model is not None
    model = args.model
    model.to(device)

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if use_cfg:
        model_fn = ema.forward_with_cfg
    else:
        model_fn = ema.forward

    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    ) # default: velocity
    transport_sampler = Sampler(transport)  
    
    # training process
    # global_step = 0
    log_step = 0
    running_loss = 0.0
    start = time.time()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # set noisy z_0 for sampling
    best_valid_r2 = -1.0
    # Prepare models for training: (multi gpus for synced weights)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    valid_r2_curve = []
    start = time.time()
    batch_progress_bar = tqdm(range(args.training_step), desc="Training", leave=True)
    for global_step in batch_progress_bar:
    # while global_step < args.training_step:
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode

        # feedforward
        train_batch_x, train_batch_y, train_batch_n = train_generator.__next__()
        train_batch_x = torch.tensor(train_batch_x).to(device) # conditional latent features
        train_batch_y = torch.tensor(train_batch_y).to(device) # expected stable neural manifold

        # half
        sample_num = train_batch_x.shape[0]
        sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1
        train_batch_x = train_batch_x[:sample_num]
        train_batch_y = train_batch_y[:sample_num]

        # compute expected neural manifold (flow destination)
        with torch.no_grad():
            exp_z_manifold = model.linear_encoder(train_batch_y)
            # resize for DM
            exp_z_manifold = torch.unsqueeze(exp_z_manifold, dim=1)

        model_kwargs = dict(y=train_batch_x)
        loss_dict = transport.training_losses(model, exp_z_manifold, model_kwargs)
        loss = loss_dict["loss"].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema, model, decay=args.ema_decay)

        # validation
        global_step += 1
        log_step += 1
        running_loss += loss.item()
        if global_step % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                # compute r2_score on test spikes
                # inverse weights (d_model * d_pos)
                pinv_decoder = torch.linalg.pinv(model.linear_encoder.weight.t())
                pred_z_manifold = torch.squeeze(loss_dict['pred'])
                dec_out_train = (pred_z_manifold - model.linear_encoder.bias) @ pinv_decoder

                r2_score_train_tmp = r2_score(train_batch_y.cpu().detach().numpy(), dec_out_train.cpu().detach().numpy())
                # print("r2 score on training dataset: %.4f" % r2_score_train_tmp)

        if global_step % args.sample_every == 0:
            # validation: sampling
            ema.eval()
            sample_fn = transport_sampler.sample_ode(num_steps=args.num_sampling_steps, sampling_method=args.sampling_method) # default to ode sampling (sampling method)

            x, y = valid_generator[0], valid_generator[1]
            valid_batch_x = torch.tensor(x).to(device)
            valid_batch_y = torch.tensor(y).to(device)

            # half
            sample_num = valid_batch_x.shape[0]
            sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1

            valid_batch_x = valid_batch_x[:sample_num]
            valid_batch_y = valid_batch_y[:sample_num]

            # noisy latent features
            z_0 = torch.randn(sample_num, model.hidden_size, device=device)
            z_0 = torch.unsqueeze(z_0, dim=1)

            with torch.no_grad():
                model.eval()
                if use_cfg:
                    sample_model_kwargs = dict(y=valid_batch_x, cfg_scale=args.cfg_scale)
                else:
                    sample_model_kwargs = dict(y=valid_batch_x)
                samples = sample_fn(z_0, model_fn, **sample_model_kwargs)[-1]
                samples = torch.squeeze(samples)

                # decoding behavior labels
                # inverse weights (d_model * d_pos)
                pinv_decoder = torch.linalg.pinv(ema.linear_encoder.weight.t())
                dec_out_valid = (samples - ema.linear_encoder.bias) @ pinv_decoder

                r2_score_valid_tmp = r2_score(valid_batch_y.cpu().detach().numpy(), dec_out_valid.cpu().detach().numpy())
                # print("r2 score on validation dataset: %.4f" % r2_score_valid_tmp)
                batch_progress_bar.set_postfix({'valid_r2': r2_score_valid_tmp, 'train_r2': r2_score_train_tmp, 'loss': running_loss / log_step})
                valid_r2_curve.append(r2_score_valid_tmp)

                if r2_score_valid_tmp > best_valid_r2:
                    best_valid_r2 = r2_score_valid_tmp
                    best_valid_model = model
            
    end = time.time()
    total_train_time = (end - start)
    train_time_per_epoch = total_train_time / global_step
    
    # save other variables
    pre_train_results = {
        'total_train_time': total_train_time,
        'train_time_per_epoch': train_time_per_epoch,
    }

    return (best_valid_r2, r2_score_valid_tmp, valid_r2_curve), best_valid_model, pre_train_results

def train_dynamic_embedder(args):
    device = args.device    
    assert args.train_generator is not None
    train_generator = args.train_generator
    assert args.valid_generator is not None
    valid_generator = args.valid_generator
    assert args.model is not None
    model = args.model
    model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    global_step = 0
    test_per_epoch = 20
    
    best_r2_score = -1.0
    batch_progress_bar = tqdm(range(args.training_step), desc="Training", leave=True)  
    for global_step in batch_progress_bar:
        model.train()
        # feedforward
        train_batch_x, train_batch_y, _ = train_generator.__next__()
        train_batch_x = torch.tensor(train_batch_x).to(device)
        train_batch_y = torch.tensor(train_batch_y).to(device)

        if args.invert_flag:
            train_batch_x = train_batch_x.permute(0, 2, 1)

        _, cond_vec = model(
            x_enc=train_batch_x,
            x_mark_enc=None,
        )

        with torch.no_grad():
            exp_z_manifold = model.linear_encoder(train_batch_y)

        # loss
        mse_loss_func = torch.nn.MSELoss()
        # mse_loss = mse_loss_func(dec_out, train_batch_y)
        mse_loss = mse_loss_func(cond_vec, exp_z_manifold)

        if global_step % test_per_epoch == 0:
            model.eval()

            with torch.no_grad():
                # compute r2_score on test spikes
                x, y = valid_generator[0], valid_generator[1]
                valid_batch_x = torch.tensor(x).to(device)
                valid_batch_y = torch.tensor(y).to(device)
                
                if args.invert_flag:
                    valid_batch_x = valid_batch_x.permute(0, 2, 1)

                _, cond_vec_test = model(                 
                    x_enc=valid_batch_x,
                    x_mark_enc=None
                )
                # inverse weights (d_model * d_pos)
                pinv_decoder = torch.linalg.pinv(model.linear_encoder.weight.t())
                dec_out_test = (cond_vec_test - model.linear_encoder.bias) @ pinv_decoder

                r2_score_tmp = r2_score(valid_batch_y.cpu().detach().numpy(), dec_out_test.cpu().detach().numpy())

                batch_progress_bar.set_postfix({'valid_r2': r2_score_tmp, 'mse_loss': mse_loss.item()}) 
                if r2_score_tmp > best_r2_score:
                    best_r2_score = r2_score_tmp 

        if args.update_flag:
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

        global_step += 1

    print("conditional best r2_score: %.4f" % best_r2_score)

    return model