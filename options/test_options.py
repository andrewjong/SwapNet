import argparse

from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self, **defaults):
        super().__init__()
        self.is_train = False
        parser = self._parser

        parser.set_defaults(max_dataset_size=50, shuffle_data=False)
        parser.add_argument(
            "--interval",
            metavar="N",
            default=1,
            type=int,
            help="only run every n images",
        )
        parser.add_argument(
            "--warp_checkpoint",
            help="Use this to run the warp stage. Specifies the checkpoint file of "
            "warp stage model, containing args.json file in same dir",
        )
        parser.add_argument(
            "--texture_checkpoint",
            help="Use this to run the texture stage. Specifies the checkpoint dir of "
            "texture stage containing args.json file",
        )
        parser.add_argument(
            "--checkpoint",
            help="Shorthand for both warp and texture checkpoint to use the 'latest' "
                 "generator file (or specify using --load_epoch). This should be the "
                 "root dir containing warp/ and texture/ checkpoint folders.",
        )
        parser.add_argument(
            "--body_dir",
            help="Directory to use as target bodys for where the cloth will be placed "
            "on. If same directory as --cloth_root, use --shuffle_data to achieve "
            "clothing transfer. If not provided, will uses --dataroot/body",
        )
        parser.add_argument(
            "--cloth_dir",
            help="Directory to use for the clothing source. If same directory as "
            "--body_root, use --shuffle_data to achieve clothing transfer. If not "
            "provided, will use --dataroot/cloth",
        )
        parser.add_argument(
            "--texture_dir",
            help="Directory to use for the clothing source. If same directory as "
            "--body_root, use --shuffle_data to achieve clothing transfer. If not "
            "provided, will use --dataroot/texture",
        )
        parser.add_argument(
            "--results_dir",
            default="results",
            help="folder to output intermediate and final results",
        )
        parser.add_argument(
            "--skip_intermediates",
            action="store_true",
            help="choose not to save intermediate cloth visuals as images for warp "
            "stage (instead, just save .npz files)",
        )

        parser.add_argument(
            "--dataroot",
            required=False,
            help="path to dataroot if cloth, body, and texture not individually specified",
        )
        # remove arguments
        parser.add_argument(
            "--model", help=argparse.SUPPRESS
        )  # remove model as we restore from checkpoint
        parser.add_argument("--name", default="", help=argparse.SUPPRESS)

        parser.set_defaults(**defaults)

    @staticmethod
    def _validate(opt):
        super(TestOptions, TestOptions)._validate(opt)

        if not (opt.body_dir or opt.cloth_dir or opt.texture_dir or opt.dataroot):
            raise ValueError(
                "Must either (1) specify --dataroot, or (2) --body_dir, --cloth_dir, "
                "and --texture_dir individually"
            )

        if not opt.dataroot:
            if opt.warp_checkpoint and not opt.body_dir:
                raise ValueError("Warp stage must have body_dir")
            if opt.texture_checkpoint and not opt.texture_dir:
                raise ValueError("Texture stage must have texture_dir")

        if not opt.warp_checkpoint and not opt.texture_checkpoint:
            raise ValueError("Must set either warp_checkpoint or texture_checkpoint")
