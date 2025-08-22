class AlignConfig:
    def __init__(
        self,
        cfg_scale=1.0,
        num_sampling_steps=50,
        aligner_method='mmd',
        mmd_loss=None,

        device='cpu',
        src_train_generator=None,
        train_generator=None,
        valid_generator=None,
        test_generator=None,
        model = None,
        pre_train_config=None,

        tgt_train_ratio=0.8,
        fine_tuning_step=10,
        learning_rate=5e-4, # learning rate
        weight_decay=1e-5, # weight decay
        sample_every=20,
        ema_decay=0.9999,
        sampling_method="dopri5",
        trans_flag=True,
    ):
        super(AlignConfig, self).__init__()
        self.cfg_scale = cfg_scale
        self.num_sampling_steps = num_sampling_steps
        self.aligner_method = aligner_method
        self.mmd_loss = mmd_loss

        self.device = device
        self.src_train_generator = src_train_generator
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.test_generator = test_generator
        self.model = model
        self.pre_train_config = pre_train_config

        self.tgt_train_ratio = tgt_train_ratio
        self.fine_tuning_step = fine_tuning_step
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sample_every = sample_every
        self.ema_decay = ema_decay
        self.sampling_method = sampling_method
        self.trans_flag = trans_flag