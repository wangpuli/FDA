import torch
import numpy as np

# one-step euler normalizing flow
class EulerLogEstimator:
  def __init__(self, model, z_0, valid_batch_x, valid_batch_y, sample_fn, likeli_fn, cfg_scale=1.0, num_sampling_steps=2, sampling_method='euler'):
    self.model = model
    self.z_0 = z_0
    self.valid_batch_x = valid_batch_x
    self.valid_batch_y = valid_batch_y
    self.sample_fn = sample_fn
    self.likeli_fn = likeli_fn
    self.cfg_scale = cfg_scale
    self.use_cfg = cfg_scale > 1.0
    self.model_fn = self.model.forward_with_cfg if self.use_cfg else self.model.forward
    self.num_sampling_steps = num_sampling_steps
    self.sampling_method = sampling_method
    self.z_t = None

  def f_transform(self, z_0, valid_batch_x):
    if len(z_0.shape) == 1:
      z_0 = torch.unsqueeze(z_0, dim=0)
    if len(z_0.shape) == 2:
      z_0 = torch.unsqueeze(z_0, dim=0)
    
    if len(valid_batch_x.shape) == 2:
      valid_batch_x = torch.unsqueeze(valid_batch_x, dim=0)
      
    device = z_0.device
    sample_num = z_0.shape[0]

    
    # naive euler
    t0, t1 = 0.0, 1.0
    t_inter = torch.linspace(t0, t1, self.num_sampling_steps)
    z_tmp = z_0
    for ti in range(self.num_sampling_steps-1):
      t = torch.ones(sample_num)*t_inter[ti]
      t = t.to(device)
      vel_pred = self.model(z_0, t, valid_batch_x)
      z_t = z_tmp + vel_pred
      z_tmp = z_t
    
    z_t_sq = z_t.squeeze()

    return z_t_sq

  def get_det_gradient(self):
    loss_value = None
    # general sampling function
    if self.use_cfg:
      sample_model_kwargs = dict(y=self.valid_batch_x, cfg_scale=self.cfg_scale)
    else:
      sample_model_kwargs = dict(y=self.valid_batch_x)
    logp, _ = self.likeli_fn(self.z_0, self.model_fn, **sample_model_kwargs)

    loss_value = -1*torch.mean(logp)
    
    return loss_value
  
  '''
  calculate the prior log likelihood
  '''
  def prior_logp(self):
      '''
          Standard multivariate normal prior
          Assume z is batched
      '''
      z = self.z_0
      shape = torch.tensor(z.size())
      N = torch.prod(shape[1:])
      _fn = lambda x: -N / 2. * np.log(2 * np.pi) - torch.sum(x ** 2) / 2.
      # piror log likelihood
      return torch.vmap(_fn)(z)