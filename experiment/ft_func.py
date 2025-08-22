import time
import torch
import numpy as np

from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy
from sklearn.metrics import r2_score
from flow.transport.transport import create_transport, Sampler
from align.likelihood import EulerLogEstimator

#################################################################################
#                                  Fine-tuning Loop                             #
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

def fine_tuning_stage(args):
    # classifier-free guidance
    use_cfg = args.cfg_scale > 1.0

    device = args.device
    assert args.src_train_generator is not None
    src_train_generator = args.src_train_generator
    assert args.train_generator is not None
    train_generator = args.train_generator
    assert args.valid_generator is not None
    valid_generator = args.valid_generator
    assert args.test_generator is not None
    test_generator = args.test_generator
    model = args.model
    model.to(device)

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if use_cfg:
        model_fn = model.forward_with_cfg
    else:
        model_fn = model.forward
    
    pre_train_config = args.pre_train_config
    transport = create_transport(
        path_type=pre_train_config.path_type,
        prediction=pre_train_config.prediction,
        loss_weight=pre_train_config.loss_weight,
        train_eps=pre_train_config.train_eps,
        sample_eps=pre_train_config.sample_eps,
    )
    transport_sampler = Sampler(transport)
    sample_fn = transport_sampler.sample_ode(num_steps=args.num_sampling_steps, sampling_method=args.sampling_method) # default to ode sampling (sampling method)
    likeli_fn = transport_sampler.sample_ode_likelihood(num_steps=args.num_sampling_steps, sampling_method=args.sampling_method) # default to ode sampling (sampling method)

    # fine-tuning process
    # set optimizer (fine-tuning conditional generators)
    optimizer = torch.optim.Adam(model.dynamic_embedder.transformer_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_valid_r2 = -1.0
    # Prepare models for training: (multi gpus for synced weights)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    best_valid_model = None
    batch_progress_bar = tqdm(range(args.fine_tuning_step), desc="Fine-tuning", leave=True)
    for global_step in batch_progress_bar: 
        model.train()  # important! This enables embedding dropout for classifier-free guidance
        ema.eval()  # EMA model should always be in eval mode
        
        train_batch_x, train_batch_y, train_batch_n = train_generator.__next__()
        train_batch_x = torch.tensor(train_batch_x).to(device)
        train_batch_y = torch.tensor(train_batch_y).to(device)

        src_batch_x, src_batch_y, src_batch_n = src_train_generator.__next__()
        src_batch_x = torch.tensor(src_batch_x).to(device)
        src_batch_y = torch.tensor(src_batch_y).to(device)

        if train_batch_n < 10 or src_batch_n < 10:
            continue

        if train_batch_n != src_batch_n and args.aligner_method == 'mmd':
            batch_num = min(train_batch_n, src_batch_n)
            train_batch_x, train_batch_y = train_batch_x[:batch_num], train_batch_y[:batch_num]
            src_batch_x, src_batch_y = src_batch_x[:batch_num], src_batch_y[:batch_num]

        batch_num = train_batch_x.shape[0]

        # noisy latent features
        z_0 = torch.randn(batch_num, model.hidden_size, device=device)
        z_0 = torch.unsqueeze(z_0, dim=1)

        align_loss = 0.0
        if args.aligner_method == 'likelihood':
            # log-likelihood estimation via normalizing flow
            logp_estimator = EulerLogEstimator(
                model=model,
                z_0=z_0,
                valid_batch_x=train_batch_x,
                valid_batch_y=train_batch_y,
                sample_fn=sample_fn,
                likeli_fn=likeli_fn,
                cfg_scale=args.cfg_scale,
                num_sampling_steps=args.num_sampling_steps,
            )
            align_loss = logp_estimator.get_det_gradient()
        elif args.aligner_method == 'mmd':
            # MMD loss
            mmd_loss = args.mmd_loss

            # calculate mmd loss on sampled latent features
            # target sampling
            if use_cfg:
                sample_model_kwargs = dict(y=train_batch_x, cfg_scale=args.cfg_scale)
            else:
                sample_model_kwargs = dict(y=train_batch_x)

            samples = sample_fn(z_0, model_fn, **sample_model_kwargs)[-1]
            samples_tgt = torch.squeeze(samples)
            # source sampling
            if use_cfg:
                sample_model_kwargs = dict(y=src_batch_x, cfg_scale=args.cfg_scale)
            else:
                sample_model_kwargs = dict(y=src_batch_x)

            samples = sample_fn(z_0, model_fn, **sample_model_kwargs)[-1]
            samples_src = torch.squeeze(samples)
            align_loss = mmd_loss(samples_src, samples_tgt) 
        else:
            raise ValueError("Invalid aligner method: %s" % args.aligner_method)

        if torch.isnan(align_loss):
            print("NaN detected in the variable align_loss")
            break
        
        # validation phase
        with torch.no_grad():
            x, y, batch_num = valid_generator[0], valid_generator[1], valid_generator[0].shape[0]
            valid_batch_x = torch.tensor(x).to(device)
            valid_batch_y = torch.tensor(y).to(device)

            # noisy latent features
            z_0 = torch.randn(batch_num, model.hidden_size, device=device)
            z_0 = torch.unsqueeze(z_0, dim=1)

            if use_cfg:
                sample_model_kwargs = dict(y=valid_batch_x, cfg_scale=args.cfg_scale)
            else:
                sample_model_kwargs = dict(y=valid_batch_x)
            samples = sample_fn(z_0, model_fn, **sample_model_kwargs)[-1]
            samples = torch.squeeze(samples)

            # decoding behavior labels
            # inverse weights (d_model * d_pos)
            pinv_decoder = torch.linalg.pinv(model.linear_encoder.weight.t())
            dec_out_valid = (samples - model.linear_encoder.bias) @ pinv_decoder

            r2_score_valid_tmp = r2_score(valid_batch_y.cpu().detach().numpy(), dec_out_valid.cpu().detach().numpy())
            batch_progress_bar.set_postfix({'align_loss': align_loss.item(), 'valid_r2': r2_score_valid_tmp})

            if r2_score_valid_tmp > best_valid_r2:
                best_valid_r2 = r2_score_valid_tmp
                # deep copy the best model
                best_valid_model = deepcopy(model).to(device)

        optimizer.zero_grad()
        align_loss.backward()
        optimizer.step()
        update_ema(ema, model, decay=args.ema_decay)

    # test phase
    # sampling
    best_valid_model.eval()
    if use_cfg:
        test_model_fn = best_valid_model.forward_with_cfg
    else:
        test_model_fn = best_valid_model.forward

    x, y, batch_num = test_generator[0], test_generator[1], test_generator[0].shape[0]
    test_batch_x = torch.tensor(x).to(device)
    test_batch_y = torch.tensor(y).to(device)

    sample_num = test_batch_x.shape[0]

    # noisy latent features
    z_0 = torch.randn(sample_num, best_valid_model.hidden_size, device=device)
    z_0 = torch.unsqueeze(z_0, dim=1)   

    with torch.no_grad():
        if use_cfg:
            sample_model_kwargs = dict(y=test_batch_x, cfg_scale=args.cfg_scale)
        else:
            sample_model_kwargs = dict(y=test_batch_x)
        samples = sample_fn(z_0, test_model_fn, **sample_model_kwargs)[-1]
        samples = torch.squeeze(samples)

        # decoding behavior labels
        # inverse weights (d_model * d_pos)
        pinv_decoder = torch.linalg.pinv(best_valid_model.linear_encoder.weight.t())
        dec_out_test = (samples - best_valid_model.linear_encoder.bias) @ pinv_decoder

        r2_score_test = r2_score(test_batch_y.cpu().detach().numpy(), dec_out_test.cpu().detach().numpy())
        print("test r2 score: %.4f" % r2_score_test)

    return (best_valid_r2, r2_score_valid_tmp, r2_score_test), best_valid_model