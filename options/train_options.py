from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.is_train = True
        parser = self._parser
        # override the model arg from base options, such that model is REQUIRED
        parser.add_argument(
            "--model",
            help="which model to run",
            choices=("warp", "texture", "pix2pix"),
            required=True
        )
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training from latest checkpoint",
        )
        # visdom and HTML visualization parameters
        parser.add_argument(
            "--display_freq",
            type=int,
            default=400,
            help="frequency of showing training results on screen",
        )
        parser.add_argument(
            "--display_ncols",
            type=int,
            default=4,
            help="if positive, display all images in a single visdom web panel with "
            "certain number of images per row.",
        )
        parser.add_argument(
            "--display_id", type=int, default=1, help="window id of the web display"
        )
        parser.add_argument(
            "--display_server",
            type=str,
            default="http://localhost",
            help="visdom server of the web display",
        )
        parser.add_argument(
            "--display_env",
            type=str,
            default="main",
            help='visdom display environment name (default is "main")',
        )
        parser.add_argument(
            "--display_port",
            type=int,
            default=8097,
            help="visdom port of the web display",
        )
        parser.add_argument(
            "--update_html_freq",
            type=int,
            default=1000,
            help="frequency of saving training results to html",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--no_html",
            action="store_true",
            help="do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        # Training parameters
        parser.add_argument(
            "--n_epochs", "--num_epochs", default=20, type=int, help="number of epochs to train until"
        )
        parser.add_argument(
            "--start_epoch", type=int, default=0, help="epoch to start training from"
        )
        parser.add_argument(
            "--sample_freq",
            help="how often to sample and save image results from the generator",
        )
        parser.add_argument(
            "--checkpoint_freq",
            default=2,
            type=int,
            help="how often to save checkpoints. negative numbers for middle of epoch",
        )
        parser.add_argument(
            "--latest_checkpoint_freq",
            default=5120,
            type=int,
            help="how often (in iterations) to save latest checkpoint",
        )
        parser.add_argument(
            "--save_by_iter",
            action="store_true",
            help="whether saves model by iteration",
        )
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
            default="kaiming",
            choices=("normal", "xavier", "kaiming"),
            help="weights initialization method",
        )
        parser.add_argument(
            "--init_gain", default=0.02, type=float, help="init scaling factor"
        )
