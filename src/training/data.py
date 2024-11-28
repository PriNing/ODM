import os
import sys
import math
import logging
import functools
import random
import pdb
import json
import pickle

import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from src.clip import transforms as T
import csv
import cv2

# from clip.clip import tokenize

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, first_stage, img_key, caption_key, char_dict_pth, gt_dir, sep="\t", 
                 single_text=False, text_batch_size=256, vocab_size=40000, context_length=77, image_resolution=512, is_train=True):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, quoting=csv.QUOTE_NONE)
        df = df.dropna()
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.bboxs = df['rotate_box'].tolist()
        self.gt_dir = gt_dir
        # self.width, self.hight = 8, 8
        # self.mask_ratio = 0.3
        
        self.transforms = transforms

        self.single_text = single_text
        self.vocab_size = vocab_size
        self.text_batch_size = text_batch_size
        
        with open(char_dict_pth, 'rb') as f:
            self.letters = pickle.load(f)
            self.letters = [chr(x) for x in self.letters]

        self.p2idx = {p: idx+1 for idx, p in enumerate(self.letters)}
        self.idx2p = {idx+1: p for idx, p in enumerate(self.letters)}

        # self.idx_mask = len(self.letters) + 1
        self.EOS = len(self.letters) + 1
        
        self.image_resolution = image_resolution
        self.first_stage = first_stage

        self.max_len = 32
        self.word_len = 25
        self.resize_img = 512
        self.use_mim = True
        self.context_length = context_length

        logging.debug('Done loading data.')

    def tokenize(self, text):
        token = np.zeros(self.word_len)
        for i in range(min(len(text), self.word_len)):
            if text[i] in self.letters:
                token[i] = self.p2idx[text[i]]
        if len(text) >= self.word_len:
            token[-1] = self.EOS
        else:
            token[len(text)] = self.EOS
        return token

    def resize_box(self, bbox_ori, ratio_w, ratio_h):

        boxx = bbox_ori.copy()
        boxx[0] = int(int(boxx[0]) * (ratio_w))
        boxx[1] = int(int(boxx[1]) * (ratio_h))
        boxx[2] = max(1, int(int(boxx[2]) * (ratio_w)))
        boxx[3] = max(1, int(int(boxx[3]) * (ratio_h)))
        return boxx

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # idx = 5
        img = np.array(Image.open(str(self.images[idx])).convert('RGB'))
        
        gt = np.array(Image.open(os.path.join(self.gt_dir, str(self.images[idx]).split('/')[-1])))

        all_texts = self.captions[idx].split(' ')
        
        all_bbox = np.array(eval(self.bboxs[idx]))

        texts = torch.zeros((self.max_len, self.word_len))
        boxes = []
        labels = []
        masked_gt = np.zeros(img.shape[:2])
        
        samList = random.sample(range(min(len(all_texts), self.max_len)), max(1, round(min(len(all_texts), self.max_len) * random.uniform(0.1, 0.3))))
        addList = random.sample([item for item in range(min(len(all_texts), self.max_len)) if item not in samList], max(0, round(min(len(all_texts), self.max_len) * random.uniform(0.1, 0.3))))
        
        flag = 1 if random.random() < 0.5 else 0

        for i in range(min(len(all_texts), self.max_len)):
            
            # if flag == 1:
            #     break
            # image mask            
            bbox = all_bbox[i]
            bbox = bbox.astype(np.int).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(bbox)
            H, W = img.shape[:2]
            if x >= W - 5 or y >= H - 5 or x < -5 or y < -5:
                continue
            if w <= 0 or h <= 0:
                continue
            
            x = max(0, x)
            y = max(0, y)
            if x + w >= W - 1:
                w = W - x - 1
            if h + y >= H - 1:
                h = H - y - 1                

            # if flag == 1 and i in samList:
            #     gt[y:y+h,x:x+w] = 0
            #     continue
                
            # if i in maskList:
            #     mask_rd = np.ones((h, w), dtype=np.uint8)
            #     mask_rd[np.random.rand(h, w) < 0.6] = 0
            #     img[y:y+h,x:x+w, :] = img[y:y+h,x:x+w, :] * np.expand_dims(mask_rd, axis=-1)
            #     masked_gt[y:y+h,x:x+w] = 1 - mask_rd

                # # masked_gt[y:y+h,x:x+w] = 1
                # mask_ids = random.sample(range(64), 16)
                # xlen = w // 8
                # ylen = h // 8
                # for idx in range(64):
                #     if idx in mask_ids:
                #         yi = idx // 8 * ylen + y
                #         xi = idx % 8 * xlen + x
                #         img[yi:yi+ylen, xi:xi+xlen, :] = 0
                #         masked_gt[yi:yi+ylen, xi:xi+xlen] = 1
            
            boxes.append([x, y, x + w, y + h])

            # text mask
            t = self.tokenize(all_texts[i])
            labels.append(t)
        
        target = dict()
        target['bboxes'] = torch.tensor(np.array(boxes).reshape(-1, 4), dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.long)

        
        img = Image.fromarray(img)    
        gt = Image.fromarray(gt)
        
        images, target, gt = self.transforms(img, target, gt)
    
        for i in range(min(len(target['labels']), self.max_len)):
            texts[i] += target['labels'][i]
        
        # if flag == 0:
        #     stt = min(len(target['labels']), self.max_len)
        #     # addList = [1]
        #     for i in range(0, min(self.max_len - stt, len(addList))):
        #         text = all_texts[addList[i]]
        #         # text = 'The'
        #         token = torch.zeros(self.word_len)
        #         for j in range(min(len(text), self.word_len)):
        #             token[j] = (self.p2idx[text[j]] + 3) % len(self.letters)
        #         if len(text) >= self.word_len:
        #             token[-1] = self.EOS
        #         else:
        #             token[len(text)] = self.EOS
        #         texts[stt + i] += token
        
        
        gt = np.array(gt)

        # gt = cv2.resize(gt, (self.resize_img, self.resize_img))

        masked_gt = cv2.resize(masked_gt, (self.resize_img, self.resize_img))

        gt_ori = gt.copy()
        gt_ori = gt_ori / 255.0

        gt[gt < 30] = 0
        gt[gt != 0] = 1
        
        # image masks can be used to mask out the padding regions during training
        image_masks = torch.zeros((self.image_resolution // 32, self.image_resolution // 32), dtype=torch.bool)

        return images, texts.long(), gt, gt_ori, image_masks, masked_gt



@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def build_transform(is_train, args):
    transforms = []
    if is_train:
        transforms.append(T.RandomCrop(args.crop_min_size_ratio, args.crop_max_size_ratio, args.crop_prob))
        transforms.append(T.RandomRotate(args.rotate_max_angle, args.rotate_prob))
        transforms.append(T.RandomResize(512, 512))
        transforms.append(T.RandomDistortion(args.dist_brightness, args.dist_contrast, args.dist_saturation, args.dist_hue, args.distortion_prob))
    else:
        transforms.append(T.RandomResize(512, 512))
        # pass
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize())

    transforms = T.Compose(transforms) if len(transforms) > 0 else None 
    return transforms


def get_csv_dataset(args, preprocess_fn, is_train):
    preprocess_fn = build_transform(is_train, args)

    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        first_stage=args.first,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        char_dict_pth=args.char_dict_pth,
        sep=args.csv_separator,
        gt_dir=args.gt_dir,
        image_resolution=args.image_size,
        is_train=is_train)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if is_train and args.distributed else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        persistent_workers=True,
        prefetch_factor=2,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

    
def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns

    data = {}

    if args.train_data:
        data["train"] = get_csv_dataset(args, preprocess_train, is_train=True)

    return data
