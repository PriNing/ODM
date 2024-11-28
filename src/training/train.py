

import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

import sys
import pdb

import logging
import random
import torchvision.transforms as T

import lpips
from src.clip.vgg import vgg16
from src.losses.lpips import OCR_CRAFT_LPIPS
from src.utils.loss import SigLipLoss

def is_master(args):
    return (not args.distributed) or args.gpu == 0


def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

class SoftDiceLoss(nn.Module): 
    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
 
    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt, activation=self.activation)

def get_loss(model, images, texts, gts, image_masks, masked_, loss_img, loss_txt, loss_mim, args):
    
    output_size = gts.shape[-2:]
    image_features, text_features, image_logits, logit_scale = model(images, texts, image_masks, output_size)

    logit_scale = logit_scale.mean()
    
    img_loss = loss_mim(image_logits, gts.unsqueeze(1).float())

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        gattered_img_loss = [
            torch.zeros_like(img_loss) for _ in range(world_size)
        ]

        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gattered_img_loss, img_loss)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )
        all_img_loss = img_loss + \
            torch.sum(torch.tensor(gattered_img_loss[:rank])).to(images.device) + \
            torch.sum(torch.tensor(gattered_img_loss[rank + 1 :])).to(images.device)

        
        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        # @为python3.5之后的矩阵乘法运算符
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        all_img_loss = img_loss


    ground_truth = torch.arange(len(logits_per_image)).long()

    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    img_loss = all_img_loss * args.img_loss_weight

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2
    
    return total_loss, img_loss

def get_loss_with_LPIPS(model, images, texts, gts, gts_ori, image_masks, loss_img, loss_txt, loss_mim, loss_lp, args):
    
    output_size = gts.shape[-2:]
    image_features, text_features, image_logits, logit_scale = model(images, texts, image_masks, output_size)

    logit_scale = logit_scale.mean()
    img_loss = loss_mim(image_logits, gts.unsqueeze(1).float())
    
    fea_loss = loss_lp(torch.sigmoid(image_logits), gts_ori.unsqueeze(1).float(), normalize=True).mean()
  

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        gattered_img_loss = [
            torch.zeros_like(img_loss) for _ in range(world_size)
        ]
        gattered_fea_loss = [
            torch.zeros_like(fea_loss) for _ in range(world_size)
        ]

        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gattered_img_loss, img_loss)
        dist.all_gather(gattered_fea_loss, fea_loss)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )
        all_img_loss = img_loss + \
            torch.sum(torch.tensor(gattered_img_loss[:rank])).to(images.device) + \
            torch.sum(torch.tensor(gattered_img_loss[rank + 1 :])).to(images.device)
        
        all_fea_loss = fea_loss + \
            torch.sum(torch.tensor(gattered_fea_loss[:rank])).to(images.device) + \
            torch.sum(torch.tensor(gattered_fea_loss[rank + 1 :])).to(images.device)

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        # @为python3.5之后的矩阵乘法运算符
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        all_img_loss = img_loss
        all_fea_loss = fea_loss

    ground_truth = torch.arange(len(logits_per_image)).long()

    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    img_loss = all_img_loss * args.img_loss_weight
    fea_loss = all_fea_loss * args.lpips_loss_weight
    # return char_loss
    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2
    
    # return char_loss, total_loss, seq_loss
    return total_loss, img_loss, fea_loss


def convert_(args, masked_imgs, masked_, roi_boxs):

    if args.mask_mode == 'all':
        indices = torch.where(masked_ != 0)
        new_boxs = roi_boxs[indices]
        new_imgs = masked_imgs[indices]
        new_boxs[:, 0] = indices[0]
    else:
        new_boxs = roi_boxs
        new_imgs = masked_imgs
        new_boxs[:, 0] = torch.arange(len(new_boxs))
    # new_boxs = new_boxs.type(torch.int)
    return new_imgs, new_boxs


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_l1 = nn.L1Loss()
    
    if args.use_l1:
        loss_mim = nn.L1Loss()
    else:
        loss_mim = nn.BCEWithLogitsLoss() # reduction='none'

    # loss_dice = SoftDiceLoss()
    if args.use_LPIPS:
        if args.use_OCR_LPIPS:
            loss_lp = OCR_CRAFT_LPIPS().eval()
        else:
            loss_lp = lpips.LPIPS(net='vgg', pnet_rand=True)
            loss_lp.net = vgg16(requires_grad=False, pretrained=True, pretrained_path='/path/pretrained/vgg16-397923af.pth')
        

    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)
        loss_mim = loss_mim.cuda(args.gpu)

        if args.use_LPIPS:
            loss_lp = loss_lp.cuda(args.gpu)
        
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()
        images, texts, gts, gts_ori, image_masks, masked_ = batch
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
            gts = gts.cuda(args.gpu, non_blocking=True)
            image_masks = image_masks.cuda(args.gpu, non_blocking=True)
            gts_ori = gts_ori.cuda(args.gpu, non_blocking=True)
            masked_ = masked_.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                if args.use_LPIPS:
                    clip_loss, mim_loss, lp_loss = get_loss_with_LPIPS(model, images, texts, gts, gts_ori, image_masks, loss_img, loss_txt, loss_mim, loss_lp, args)
                    total_loss = clip_loss + mim_loss + lp_loss
                else:
                    clip_loss, mim_loss = get_loss(model, images, texts, gts, image_masks, masked_, loss_img, loss_txt, loss_mim, args)
                    total_loss = clip_loss + mim_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            if args.use_LPIPS:
                clip_loss, mim_loss, lp_loss = get_loss_with_LPIPS(model, images, texts, gts, gts_ori, image_masks, loss_img, loss_txt, loss_mim, loss_lp, args)
                total_loss = clip_loss + mim_loss + lp_loss
            else:
                clip_loss, mim_loss = get_loss(model, images, texts, gts, image_masks,  masked_, loss_img, loss_txt, loss_mim, args)
                total_loss = clip_loss + mim_loss

            total_loss.backward()
            optimizer.step()
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()
        if is_master(args) and (i % args.log_frequency) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = args.log_frequency * 1.0 * i / num_batches_per_epoch
            if args.use_LPIPS:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                    f"Loss: mim loss: {mim_loss.item():.6f}\tclip loss: {clip_loss.item():.6f}\tlp loss: {lp_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"\tLR: {optimizer.param_groups[0]['lr']:5f}"
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                    f"Loss: mim loss: {mim_loss.item():.6f}\tclip loss: {clip_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"\tLR: {optimizer.param_groups[0]['lr']:5f}"
                )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
