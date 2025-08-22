class TrainConfig:
    def __init__(
        self,
        cfg_scale=4.0,
        path_type = "Linear",
        prediction = "velocity",
        loss_weight = None,
        train_eps = None,
        num_sampling_steps = 50,
        lya_num_sampling_steps = 10,

        device='cpu',
        train_generator=None,
        valid_generator=None,
        model=None,

        training_step=1000,
        learning_rate=2e-3, # learning rate
        weight_decay=1e-5, # weight decay
        sample_every=20,
        ema_decay=0.9999,
        sampling_method="dopri5",
    ):
        super(TrainConfig, self).__init__()

        # scale of classifier-free guidance
        self.cfg_scale = cfg_scale
        # flow path configuration
        assert path_type in ["Linear", "GVP", "VP"]
        self.path_type = path_type
        assert prediction in ["velocity", "score", "noise"]
        self.prediction = prediction
        assert loss_weight in [None, "velocity", "likelihood"]
        self.loss_weight = loss_weight
        self.sample_eps = 1e-1
        self.train_eps = train_eps
        self.num_sampling_steps = num_sampling_steps
        self.lya_num_sampling_steps = lya_num_sampling_steps

        # training data
        self.device = device
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.model = model

        # training steps and optimizer
        self.training_step = training_step
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sample_every = sample_every
        self.ema_decay = ema_decay

        # sampling
        self.sampling_method = sampling_method

class TrainEmbedderConfig:
    def __init__(
        self,
        device='cpu',
        train_generator=None,
        valid_generator=None,
        model=None,

        invert_flag=False,

        training_step=1000,
        learning_rate=2e-3, # learning rate
        weight_decay=1e-5, # weight decay
        sample_every=20,
        update_flag=True,
        ):
        super(TrainEmbedderConfig, self).__init__()

        self.device = device
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.model = model

        self.invert_flag = invert_flag

        self.training_step = training_step
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sample_every = sample_every
        self.update_flag = update_flag