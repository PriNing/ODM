

import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json

import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.clip.clip import _transform, load
from src.clip.model import convert_weights, oCLIP
from src.training.train import train
from src.training.data import get_data
from src.training.params import parse_args
from src.training.logger import setup_primary_logging, setup_worker_logging
from src.training.scheduler import cosine_lr
import numpy as np


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp

def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    #args.rank = gpu
    if ngpus_per_node is None:
        ngpus_per_node = 0
    rank = args.rank * ngpus_per_node + gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )
    
    if args.dp:
        args.batch_size *= args.world_size

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    # Do not use skip_reset unless you want to use on of the CLIP model

    model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    if args.use_slip:
        model_info['init_logit_scale'] = np.log(10) 
        model_info['init_logit_bias'] = -10
    
    model = oCLIP(args.first, **model_info)
    convert_weights(model)
    args.image_size = model.visual.input_resolution
    preprocess_train = _transform(model.visual.input_resolution, is_train=True)
    preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)
    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)

    data = get_data(args, (preprocess_train, preprocess_val))

    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}                    
            model.load_state_dict(sd)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    if args.auto_resume and os.path.exists(os.path.join(args.logs, args.name)):
        folder_path = os.path.join(args.logs, args.name, 'checkpoints')
        last_epoch = None
        file_prefix = "epoch_"  # 文件前缀
        file_suffix = ".pt"  # 文件后缀
        for file in os.listdir(folder_path):
            # print("#######",file)
            if file.startswith(file_prefix) and file.endswith(file_suffix):
                epoch_str = file[len(file_prefix): -len(file_suffix)]
                # print("#######",file, epoch_str)

                try:
                    epoch = int(epoch_str)
                    # print("##########",epoch)
                    if last_epoch is None or epoch > last_epoch:
                        last_epoch = epoch
                except ValueError:
                    pass
        if last_epoch is not None:
            resume_file = file_prefix + str(last_epoch) + file_suffix
            resume_file = os.path.join(folder_path, resume_file)
            if args.gpu is None:
                checkpoint = torch.load(resume_file)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(resume_file, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}                    
            model.load_state_dict(sd)

            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                f"=> loaded checkpoint '{resume_file}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.name))

    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)    
            new_dict = dict()
            # model.load_state_dict(state_dict.state_dict())
            
            sd = checkpoint["state_dict"]
            # print('Loaded', new_dict.keys())
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            logging.info(
                f"=> loaded pretrained '{args.pretrained}')"
            )  

    # print(model.state_dict().keys())

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
        (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')
        train(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        steps = data["train"].dataloader.num_batches * (epoch + 1)

        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)):
            if (epoch + 1) == args.epochs or (
                args.save_frequency > 0 and ((epoch + 1) % args.save_frequency) == 0
            ):
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "name": args.name,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )



def main():
    args = parse_args()
    
    # get the name of the experiments
    if args.name is None:
        # args.name = strftime(
        #     f"lr={args.lr}_"
        #     f"wd={args.wd}_"
        #     f"agg={args.aggregate}_"
        #     f"model={args.model}_"
        #     f"batchsize={args.batch_size}_workers={args.workers}",
        #     # f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S",
        #     # gmtime(),
        # )
        args.name = ("{}_lr={}_wd={}_agg={}_model={}_batchsize={}_workers={}".format(args.prefix, args.lr, args.wd, args.aggregate, args.model, args.batch_size, args.workers))

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    # if os.path.exists(args.log_path):
    #     print(
    #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
    #     )
    #     return -1

    assert args.precision in ['amp', 'fp16', 'fp32']
    #assert args.model in ['RN50', 'RN101', 'RN50x4', 'ViT-B/32'] or os.path.exists(args.model)

    args.ngpus_per_node = torch.cuda.device_count()

    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node*(args.nmechine)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, args))
    else:
        if args.dp:
            args.gpu = args.multigpu[0]
            args.world_size = len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


if __name__ == "__main__":
    main()
