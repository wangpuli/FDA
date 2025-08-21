class ModelConfig:
    def __init__(
        self,
        seq_len=6,# length of input series
        pred_len=1,# length of output precited series
        output_attention=True,# flag of output attn
        use_norm=False,# flag of use_norm
        enc_in=96,# encoder input size
        d_model=64,# dimension of model
        factor=1,# attn factor
        dropout=0.1,# dropout rate
        n_heads=8, # number of heads
        d_ff=512, # dimension of hcn
        activation='gelu', # activation function
        e_layers=2, # number of encoder layers
        d_pos=2, # dimension of position
        learning_rate=2e-3, # learning rate
        weight_decay=1e-5, # weight decay
        training_step=1000, # training steps
    ):
        super(ModelConfig, self).__init__()

        # forecasting task
        self.seq_len = seq_len
        self.pred_len = pred_len

        # flag
        self.output_attention = output_attention
        self.use_norm = use_norm

        # model settings
        self.enc_in = enc_in
        self.d_model = d_model
        self.factor = factor
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.d_pos = d_pos

        # training settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_step = training_step