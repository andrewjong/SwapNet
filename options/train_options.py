from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.is_train = True
        parser = self._parser
        parser.add_argument("--continue_train",action="store_true", help="continue training from latest checkpoint")
        # Training parameters
        parser.add_argument(
            "--n_epochs", default=20, type=int, help="number of epochs to train until"
        )
        parser.add_argument(
            "--sample_freq",
            help="how often to sample and save image results from the generator",
        )
        parser.add_argument(
            "--checkpoint_freq",
            default=5,
            type=int,
            help="how often to save checkpoints. negative numbers for middle of epoch",
        )
        parser.add_argument(
            "--latest_checkpoint_freq",
            default=5120,
            type=int,
            help="how often (in iterations) to save latest checkpoint",
        )
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument(
            "--lr",
            "--learning_rate",
            type=float,
            default=0.01,
            help="initial learning rate",
        )
        parser.add_argument(
            "--wt_decay",
            "--weight_decay",
            dest="weight_decay",
            default=0,
            type=float,
            help="optimizer L2 weight decay",
        )
        # weights init
        parser.add_argument(
            "--init_type",
            default="normal",
            choices=("normal", "xavier", "kaiming"),
            help="weights initialization method",
        )
        parser.add_argument(
            "--init_gain", default=0.02, type=float, help="init scaling factor"
        )
