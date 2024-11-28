


import argparse
import pickle

def get_default_params(model_name):
    if 'ViT' in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv file with training data",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Path to mask file with training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default=None,
        help="Path to json file",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=1000
        )
    parser.add_argument(
        "--max-len",
        type=int,
        default=32
    )
    parser.add_argument(
        "--word-len",
        type=int, 
        default=25,
    )
    parser.add_argument(
        "--padding-index",
        type=int,
        default="98",
        help="The index for padding the text to a fix length."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Saved file prefix.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--first",
        default=False,
        action="store_true",
        help="If train first stage only or not",
    )
    parser.add_argument(
        "--char_loss_weight",
        type=float,
        default=1.0,
        help="Weight of character prediction loss",
    )
    parser.add_argument(
        "--img_loss_weight",
        type=float,
        default=1.0,
        help="Weight of MIM prediction loss",
    )
    parser.add_argument(
        "--seq_loss_weight",
        type=float,
        default=1.0,
        help="Weight of sequence prediction loss",
    )
    parser.add_argument(
        "--lpips_loss_weight",
        type=float,
        default=1.0,
        help="Weight of sequence prediction loss",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--log-frequency", type=int, default=100, help="How often to print logs."
    )
    parser.add_argument(
        '--auto-resume', type=bool, default=True, help="Whether to resume from the current workdir."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        type=str,
        help="path to pretrained checkpoint (default: none)",
    )
    parser.add_argument(
        "--char-dict-pth",
        default="./data/SynthText/char_dict",
        type=str,
        help="path to character dictionary",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--model",
        default="RN50",
        type=str,
        help="Name of the vision backbone to use.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--rank",
        default="0",
        type=int,
        help="rank",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['tensorboard']"
    )
    parser.add_argument(
        "--C", type=float, default=3.16, help="inverse regularizer for logistic reg."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--dp",
        default=False,
        action="store_true",
        help="Use DP instead of DDP."
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="In DP, which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--nmechine",
        default=1,
        type=int,
        help="The number of mechines used to train the model",
    )

    parser.add_argument(
        "--mask_mode",
        default='all',
        type=str,
        help="Mask Image choice",
    )
    parser.add_argument(
        "--use-LPIPS",
        action="store_true",
        default=False,
        help="use LPIPS loss",
    )
    parser.add_argument(
        "--use-OCR-LPIPS",
        action="store_true",
        default=False,
        help="use LPIPS loss",
    )
    parser.add_argument(
        "--use-l1",
        action="store_true",
        default=False,
        help="use L1 loss",
    )
    parser.add_argument(
        "--use-slip",
        action="store_true",
        default=False,
        help="use slip loss",
    )
    
    parser.add_argument('--batch_aug', action='store_true')
    parser.add_argument('--train_min_size', type=int, nargs='+', default=[640, 672, 704, 736, 768, 800, 832, 864, 896])
    parser.add_argument('--train_max_size', type=int, default=1600)
    parser.add_argument('--test_min_size', type=int, default=1000)
    parser.add_argument('--test_max_size', type=int, default=1824)
    # Data Augmentation
    parser.add_argument('--crop_min_size_ratio', type=float, default=0.5)
    parser.add_argument('--crop_max_size_ratio', type=float, default=1.0)
    parser.add_argument('--crop_prob', type=float, default=1.0)
    parser.add_argument('--rotate_max_angle', type=int, default=30)
    parser.add_argument('--rotate_prob', type=float, default=0.3)
    parser.add_argument('--dist_brightness', type=float, default=0.5)
    parser.add_argument('--dist_contrast', type=float, default=0.5)
    parser.add_argument('--dist_saturation', type=float, default=0.5)
    parser.add_argument('--dist_hue', type=float, default=0.5)
    parser.add_argument('--distortion_prob', type=float, default=0.5)
    parser.add_argument('--letters', type=str,
                        default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')

    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    # with open(args.char_dict_pth, 'rb') as f:
    #     args.letters = pickle.load(f)
    #     args.letters = [chr(x) for x in args.letters]
    
    with open(args.char_dict_pth, 'r') as f:
        args.letters = f.read().strip()
        args.letters = [item for item in args.letters]
   

    # args.num_char_classes = len(args.letters) + 1 # 1 for unkonwn
    # args.recog_pad_index = args.num_bins + args.num_char_classes
    # args.idx_mask = args.recog_pad_index + 1
    # args.padding_index = args.idx_mask + 1
    # args.SOS = args.padding_index + 1
    # args.EOS = args.SOS + 1
    # args.num_classes = args.EOS + 1
    # args.max_position_embeddings = (2 + args.word_len) * args.max_len + 1

    args.num_char_classes = len(args.letters) + 1 # 1 for unkonwn
    args.recog_pad_index = args.num_bins + args.num_char_classes
    args.EOS = args.recog_pad_index + 1
    args.SOS = args.EOS + 1
    args.padding_index = args.SOS + 1
    args.num_classes = args.padding_index + 1
    args.max_position_embeddings = (2 + args.word_len) * args.max_len + 1

    return args
