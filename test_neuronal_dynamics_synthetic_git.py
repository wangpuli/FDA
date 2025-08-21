# ---- Imports ---- #
import torch
import numpy as np 
import random
import logging
logger = logging.getLogger(__name__)

from scipy.special import gammaln
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
# ---- For Simulated Neural Data ---- #
from ldns.data.latent_attractor import get_attractor_dataloaders
from model.vanilla_iTransfomer import ConditionModel
from config.model_config import ModelConfig
from flow.models.SiT_models import SiT
from flow.transport.transport import create_transport, Sampler

# set seed
def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

# evaluate co-bps
def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
        spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

# compute bits-per-spike
def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

# generate dataloader
def load_data_loader(spikes, rates, latents, window_size, is_shuffle=True, is_batch=True):
    rates_win, latents_win, spike_win = None, None, None
    for i in range(rates.shape[0]):
        for j in range(window_size-1, rates.shape[1]):
            if rates_win is None:
                rates_win = rates[i:i+1, j-window_size+1:j+1, :]
                latents_win = latents[i:i+1, j:j+1, :]
                spike_win = spikes[i:i+1, j-window_size+1:j+1, :]
            else:
                rates_win = torch.cat((rates_win, rates[i:i+1, j-window_size+1:j+1, :]), axis=0)
                latents_win = torch.cat((latents_win, latents[i:i+1, j:j+1, :]), axis=0)
                spike_win = torch.cat((spike_win, spikes[i:i+1, j-window_size+1:j+1, :]), axis=0)

    dataset = TensorDataset(spike_win, rates_win, latents_win)
    if is_batch:
        dataloader = DataLoader(dataset, 
                                batch_size=256,
                                shuffle=is_shuffle,)
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=rates_win.shape[0],
                                shuffle=is_shuffle,)
    return dataloader

if __name__ == "__main__":

    final_r2_score = []

    # ---- Run params ---- #
    # make sure to match these to the config!
    # these are the default values in the paper

    system_name = "Lorenz"
    signal_length = 32 # length of each sequence
    total_in = 96  # number of neurons
    C_in = total_in // 2  # number of input channels (neurons)
    n_ic = 600  # number of initial conditions (total sequences)
    # mean_rate = 0.3  # mean firing rate in Hz
    split_frac_train = 0.7  # fraction of data for training
    split_frac_val = 0.1  # fraction of data for validation
    random_seed = 42  # for reproducibility
    softplus_beta = 2.0  # controls sharpness of rate nonlinearity
    mean_rate = 0.05  # mean firing rate in Hz

    # ---- Generate data ---- #
    # create dataloaders for train/val/test splits
    train_dataloader, val_dataloader, test_dataloader, dataset = get_attractor_dataloaders(
        system_name=system_name,
        n_neurons=total_in,
        sequence_length=signal_length,
        # noise_std=0.05,
        n_ic=n_ic,
        mean_spike_count=mean_rate * signal_length,
        train_frac=split_frac_train,
        valid_frac=split_frac_val,  # test is 1 - train - valid
        random_seed=random_seed,
        batch_size=1,
        softplus_beta=softplus_beta,
    )

    # ---- Extract data from dataloaders ---- #
    # extract spikes (shape: [batch, time, neurons])

    train_spikes = torch.stack(
        [train_dataloader.dataset[i]["signal"] for i in range(len(train_dataloader.dataset))]
    ).permute(0, 2, 1)

    val_spikes = torch.stack(
        [val_dataloader.dataset[i]["signal"] for i in range(len(val_dataloader.dataset))]
    ).permute(0, 2, 1)

    test_spikes = torch.stack(
        [test_dataloader.dataset[i]["signal"] for i in range(len(test_dataloader.dataset))]
    ).permute(0, 2, 1)

    # extract rates (shape: [batch, time, neurons])
    train_rates = torch.stack(
        [train_dataloader.dataset[i]["rates"] for i in range(len(train_dataloader.dataset))]
    ).permute(0, 2, 1)

    val_rates = torch.stack(
        [val_dataloader.dataset[i]["rates"] for i in range(len(val_dataloader.dataset))]
    ).permute(0, 2, 1)

    test_rates = torch.stack(
        [test_dataloader.dataset[i]["rates"] for i in range(len(test_dataloader.dataset))]
    ).permute(0, 2, 1)

    # extract latents (shape: [batch, time, latent_dim])
    train_latents = torch.stack(
        [train_dataloader.dataset[i]["latents"] for i in range(len(train_dataloader.dataset))]
    ).permute(0, 2, 1)

    val_latents = torch.stack(
        [val_dataloader.dataset[i]["latents"] for i in range(len(val_dataloader.dataset))]
    ).permute(0, 2, 1)

    test_latents = torch.stack(
        [test_dataloader.dataset[i]["latents"] for i in range(len(test_dataloader.dataset))]
    ).permute(0, 2, 1)

    # print data shapes for verification
    print(f"Train data shape: {train_spikes.shape}")
    print(f"Valid data shape: {val_spikes.shape}")
    print(f"Test data shape: {test_spikes.shape}")
    print(f"Train rates shape: {train_rates.shape}")
    print(f"Train latents shape: {train_latents.shape}")


    win_size = 4
    train_dataloader = load_data_loader(train_spikes, train_rates, train_latents, win_size, is_shuffle=False)
    val_dataloader = load_data_loader(val_spikes, val_rates, val_latents, win_size, is_shuffle=False, is_batch=False)
    test_dataloader = load_data_loader(test_spikes, test_rates, test_latents, win_size , is_shuffle=False, is_batch=False)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    invert_flag = False
    training_step = 250


    # vanilla Transformer (encoder-only)
    seed = 0
    setup_seed(seed)
    context_size = win_size
    n_chan  = train_rates[0].shape[1]//2 if not invert_flag else context_size
    seq_len = context_size if not invert_flag else train_rates[0].shape[1]
    configs = ModelConfig(
        seq_len=seq_len,
        enc_in=n_chan,
        training_step=training_step,
        e_layers=2,
        factor=1,
        n_heads=8,
        d_model=n_chan, 
    )
    transformer_model = ConditionModel(configs)

    # SiT model settings
    flow_model = SiT(
        in_channels=n_chan if not invert_flag else seq_len,
        window_size=context_size,
        hidden_size=n_chan,
        out_dim=train_latents.shape[-1],
        depth=5,
        mlp_ratio=2.0,
        num_heads=8,
        model_config=configs,
        invert_flag=invert_flag,
    )
    model_fn = flow_model.forward
    flow_model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=2e-3, weight_decay=1e-5)

    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=1e-1,
    ) # default: velocity
    transport_sampler = Sampler(transport)

    # train
    training_step = 300
    best_val_r2 = -1
    sample_fn = transport_sampler.sample_ode(num_steps=2, sampling_method="euler")
    for global_step in range(training_step):
        batch_progress_bar = tqdm(train_dataloader, desc="Training", leave=True)  
        for batch_idx, (train_batch_spikes, train_batch_rates, train_batch_latents) in enumerate(batch_progress_bar):
            flow_model.train()

            train_batch_spikes = train_batch_spikes.clone().detach().to(device)
            train_batch_rates = train_batch_rates.clone().detach().to(device)
            train_batch_latents = train_batch_latents.clone().detach().to(device)

            with torch.no_grad():
                exp_z_manifold = flow_model.linear_encoder(train_batch_latents)

            model_kwargs = dict(y=train_batch_rates[:, :, :n_chan])
            loss_dict = transport.training_losses(flow_model, exp_z_manifold, model_kwargs)
            loss = loss_dict["loss"].mean()

            # reconstruction loss
            nll_func = torch.nn.PoissonNLLLoss()
            pred_rates = torch.squeeze(flow_model.reconstruction_decoder(exp_z_manifold))
            loss += nll_func(pred_rates, train_batch_rates[:, -1, n_chan:]) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "batch": batch_idx,
            })
        
        # validation
        if (global_step+1) % 20 == 0:
            with torch.no_grad():
                flow_model.eval()
                for _, (val_batch_spikes, val_batch_rates, val_batch_latents) in enumerate(val_dataloader):
                    val_batch_spikes = val_batch_spikes.clone().detach().to(device)
                    val_batch_rates = val_batch_rates.clone().detach().to(device)
                    val_batch_latents = val_batch_latents.clone().detach().to(device)

                    # noisy latent features
                    sample_num = val_batch_rates.shape[0]
                    z_0 = torch.randn(sample_num, flow_model.hidden_size, device=device)
                    z_0 = torch.unsqueeze(z_0, dim=1)

                    sample_model_kwargs = dict(y=val_batch_rates[:, :, :n_chan])
                    samples = sample_fn(z_0, model_fn, **sample_model_kwargs)[-1]
                    samples = torch.squeeze(samples)

                    pinv_decoder = torch.linalg.pinv(flow_model.linear_encoder.weight.t())
                    dec_out_valid = (samples - flow_model.linear_encoder.bias) @ pinv_decoder

                    y_true = torch.squeeze(val_batch_latents[:sample_num]).clone().detach()
                    y_pred = dec_out_valid[:sample_num].clone().detach()

                    pred_rates = torch.squeeze(flow_model.reconstruction_decoder(samples))
                    pred_rates = pred_rates.exp().clone().detach().cpu().numpy()

                    # co-bps of reconstructed signals
                    co_bps = bits_per_spike(pred_rates, torch.squeeze(val_batch_spikes[:sample_num, -1, n_chan:]).clone().detach().cpu().numpy())
                    print("epoch: %d, val co-bps: %.4f" % (global_step+1, co_bps))

                    r2_score_val_tmp = r2_score(y_true.cpu().detach().numpy(), y_pred.clone().cpu().detach().numpy())
                    print("epoch: %d, val r2 score: %.4f" % (global_step+1, r2_score_val_tmp))

                    if r2_score_val_tmp > best_val_r2:
                        best_val_r2 = r2_score_val_tmp
                        best_val_model = deepcopy(flow_model)

    # test
    model_fn_test = best_val_model.forward
    with torch.no_grad():
        best_val_model.eval()
        for _, (test_batch_spikes, test_batch_rates, test_batch_latents) in enumerate(test_dataloader):
            test_batch_spikes = test_batch_spikes.clone().detach().to(device)
            test_batch_rates = test_batch_rates.clone().detach().to(device)
            test_batch_latents = test_batch_latents.clone().detach().to(device)

            # noisy latent features
            sample_num = test_batch_rates.shape[0]
            z_0 = torch.randn(sample_num, best_val_model.hidden_size, device=device)
            z_0 = torch.unsqueeze(z_0, dim=1)

            sample_model_kwargs = dict(y=test_batch_rates[:, :, :n_chan])
            samples = sample_fn(z_0, model_fn_test, **sample_model_kwargs)[-1]
            samples = torch.squeeze(samples)

            pinv_decoder = torch.linalg.pinv(best_val_model.linear_encoder.weight.t())
            dec_out_test = (samples - best_val_model.linear_encoder.bias) @ pinv_decoder

            y_true = torch.squeeze(test_batch_latents[:sample_num]).clone().detach()
            y_pred = dec_out_test[:sample_num].clone().detach()

            pred_rates = torch.squeeze(best_val_model.reconstruction_decoder(samples))
            pred_rates = pred_rates.exp().clone().detach().cpu().numpy()

            # co-bps of reconstructed signals
            co_bps = bits_per_spike(pred_rates, torch.squeeze(test_batch_spikes[:sample_num, -1, n_chan:]).clone().detach().cpu().numpy())
            print("seed: %d, test co-bps: %.4f" % (seed, co_bps))

            r2_score_test_tmp = r2_score(y_true.cpu().detach().numpy(), y_pred.clone().cpu().detach().numpy())
            final_r2_score.append(r2_score_test_tmp)
            print("seed: %d, test r2 score: %.4f" % (seed, r2_score_test_tmp))