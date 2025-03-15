import sys
from PIL import Image
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.transforms as T
import pickle
import json
import argparse
import torch.nn.functional as F
from torch import nn

from src.clip.clip import _transform
from src.clip.model import oCLIP

def load_model(model_path, model_info):

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict['state_dict']
    state_dict_ = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = oCLIP(False, **model_info)
    model.eval()
    model.load_state_dict(state_dict_)
    return model

def visualize(image, image_mask, preds, save_path, idx):

    att_show = image_mask[0]

    att_show = att_show[:, 1:].view(257, 16, 16)

    att_show = torch.mean(att_show, dim=0).cpu().numpy()

    im = image[0].permute(1, 2, 0).numpy()
    im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    h, w, _ = im.shape
    mask = cv2.resize(att_show, dsize=(w, h))
    mask = 255 - cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8) 

    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    added_image = cv2.addWeighted(im, 0.8, mask,0.6, 0)
    cv2.imwrite('{}/im{}_attn_demo.jpg'.format(save_path, idx), added_image[:, :, ::-1])
    cv2.imwrite('{}/{}_attn_demo.jpg'.format(save_path, idx), mask[:, :, ::-1])

    preds = torch.sigmoid(preds)[0][0]
    preds = preds.numpy() * 255
    preds = Image.fromarray(preds.astype(np.uint8)).convert('L')
    preds.save(os.path.join(save_path, 'pred_{}.png'.format(idx)))


def tokenize(char_dict_pth, text):
    with open(char_dict_pth, 'rb') as f:
        letters = pickle.load(f)
        letters = [chr(x) for x in letters]
        p2idx = {p: idx+1 for idx, p in enumerate(letters)}
        idx2p = {idx+1: p for idx, p in enumerate(letters)}
    
    word_len = 25
    token = torch.zeros(word_len)
    EOS = len(letters) + 1
    for i in range(min(len(text), word_len)):
        if text[i] in letters:
            token[i] = p2idx[text[i]]
    if len(text) >= word_len:
        token[-1] = EOS
    else:
        token[len(text)] = EOS
    return token


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visualization of ODM trained Model")

    parser.add_argument("--img_path", type=str)
    parser.add_argument("--img_text", nargs='+')

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--char_dict_pth", type=str)
    parser.add_argument("--model_config_file", type=str, default="src/training/model_configs/RN50_Seg_Clip.json")
    parser.add_argument("--save_path", type=str, default="demo/")
    
    args = parser.parse_args()

    device = "cpu"

    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)
    with open(args.model_config_file, 'r') as f:
        model_info = json.load(f)
    
    model = load_model(args.model_path, model_info)
    model.to(device)

    args.image_size = model.visual.input_resolution

    preprocess_val = _transform(model.visual.input_resolution, is_train=False)
  
    text_ = args.img_text

    image = Image.open(args.img_path)
    img_t = preprocess_val(image)

    images = img_t.unsqueeze(0)
    images = images.to(device)

    texts = torch.zeros((32, 25))
    for j in range(len(text_)):
        t = tokenize(args.char_dict_pth ,text_[j])
        texts[j] += t
    texts = texts.long().unsqueeze(0)
    texts = texts.to(device)
    image_mask = None

    with torch.no_grad():
        output_size = images.shape[-2:]
        preds, att_maps = model(images, texts, image_mask, output_size)

    visualize(images, att_maps, preds, args.save_path, args.img_path.split('/')[-1].split('.')[0])

 