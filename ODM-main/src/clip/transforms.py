import sys 
sys.path.append('.')

import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from copy import deepcopy 
from src.utils.misc import bezier2bbox


class RandomCrop(object):
    def __init__(self, min_size_ratio, max_size_ratio, prob):
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.prob = prob 

    def __call__(self, image, target, gt):
        if random.random() > self.prob or len(target['bboxes']) == 0:
            return image, target, gt

        for _ in range(100):
            crop_w = int(image.width * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_h = int(image.height * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_region = transforms.RandomCrop.get_params(image, [crop_h, crop_w])
            cropped_image, cropped_target, cropped_gt = self.crop(deepcopy(image), deepcopy(target), deepcopy(gt), crop_region)
            if not cropped_image is None:
                return cropped_image, cropped_target, cropped_gt

        print('Can not be cropped with texts')
        return image, target, gt
    
    def crop(self, image, target, gt, crop_region):
        bboxes = target['bboxes']
        crop_region, keep_instance = self.adjust_crop_region(bboxes, crop_region)
        
        if crop_region is None:
            return None, None, None

        cropped_image = F.crop(image, *crop_region)
        cropped_gt = F.crop(gt, *crop_region)

        rg_ymin, rg_xmin, rg_h, rg_w = crop_region
        # target['size'] = torch.tensor([rg_h, rg_w])
        if bboxes.shape[0] > 0:
            target['bboxes'] = target['bboxes'] - torch.tensor([rg_xmin, rg_ymin] * 2)
            # target['seg_pts'] = target['seg_pts'] - torch.tensor([rg_xmin, rg_ymin] * 4)
            for k in ['labels', 'bboxes']:
                target[k] = target[k][keep_instance]

        return cropped_image, target, cropped_gt

    def adjust_crop_region(self, bboxes, crop_region):
        rg_ymin, rg_xmin, rg_h, rg_w = crop_region 
        rg_xmax = rg_xmin + rg_w 
        rg_ymax = rg_ymin + rg_h 

        pre_keep = torch.zeros((bboxes.shape[0], ), dtype=torch.bool)
        while True:
            ov_xmin = torch.clamp(bboxes[:, 0], min=rg_xmin)
            ov_ymin = torch.clamp(bboxes[:, 1], min=rg_ymin)
            ov_xmax = torch.clamp(bboxes[:, 2], max=rg_xmax)
            ov_ymax = torch.clamp(bboxes[:, 3], max=rg_ymax)
            ov_h = ov_ymax - ov_ymin 
            ov_w = ov_xmax - ov_xmin 
            keep = torch.bitwise_and(ov_w > 0, ov_h > 0)

            if (keep == False).all():
                return None, None

            if keep.equal(pre_keep):
                break 

            keep_bboxes = bboxes[keep]
            keep_bboxes_xmin = int(min(keep_bboxes[:, 0]).item())
            keep_bboxes_ymin = int(min(keep_bboxes[:, 1]).item())
            keep_bboxes_xmax = int(max(keep_bboxes[:, 2]).item())
            keep_bboxes_ymax = int(max(keep_bboxes[:, 3]).item())
            rg_xmin = min(rg_xmin, keep_bboxes_xmin)
            rg_ymin = min(rg_ymin, keep_bboxes_ymin)
            rg_xmax = max(rg_xmax, keep_bboxes_xmax)
            rg_ymax = max(rg_ymax, keep_bboxes_ymax)

            pre_keep = keep
        
        crop_region = (rg_ymin, rg_xmin, rg_ymax - rg_ymin, rg_xmax - rg_xmin)
        return crop_region, keep


class RandomRotate(object):
    def __init__(self, max_angle, prob):
        self.max_angle = max_angle 
        self.prob = prob 

    def __call__(self, image, target, gt):
        if random.random() > self.prob or target['bboxes'].shape[0] <= 0:
            return image, target, gt
        
        angle = random.uniform(-self.max_angle, self.max_angle)
        image_w, image_h = image.size
        rotation_matrix = cv2.getRotationMatrix2D((image_w//2, image_h//2), angle, 1)
        
        image = image.rotate(angle, expand=True)
        gt = gt.rotate(angle, expand=True)

        # new_w, new_h = image.size 
        # target['size'] = torch.tensor([new_h, new_w])
        # pad_w = (new_w - image_w) / 2
        # pad_h = (new_h - image_h) / 2

        # seg_pts = target['seg_pts'].numpy()
        # seg_pts = seg_pts.reshape(-1, 4, 2)
        # seg_pts = self.rotate_points(seg_pts, rotation_matrix, (pad_w, pad_h))
        # seg_pts = seg_pts.reshape(-1, 8)
        # target['seg_pts'] = torch.from_numpy(seg_pts).type(torch.float32)
        # bboxes = [cv2.boundingRect(ele.astype(np.uint).reshape((-1, 1, 2))) for ele in seg_pts]
        


        # target['bboxes'] = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        # target['bboxes'][:, 2] += target['bboxes'][:, 0]
        # target['bboxes'][:, 3] += target['bboxes'][:, 1]

        return image, target, gt

    def rotate_points(self, coords, rotation_matrix, paddings):
        coords = np.pad(coords, (0, 0), mode='constant', constant_values=1)
        # coords = np.pad(coords, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
        coords = np.dot(coords, rotation_matrix.transpose())
        coords[:, :, 0] += paddings[0]
        coords[:, :, 1] += paddings[1]
        return coords


class RandomResize(object):
    def __init__(self, min_sizes, max_size):
        self.min_sizes = min_sizes
        self.max_size = max_size 
    
    def __call__(self, image, target, gt):
        # min_size = random.choice(self.min_sizes)
        
        # size = self.get_size_with_aspect_ratio(image.size, min_size, self.max_size)
        
        size = (512, 512)
        # size = (640, 640)
        rescaled_image = F.resize(image, size)
        rescaled_gt = F.resize(gt, size)

        ratio_width = rescaled_image.size[0] / image.size[0]
        ratio_height = rescaled_image.size[1] / image.size[1]

        # target['size'] = torch.tensor(size)
        # # target['area'] = target['area'] * (ratio_width * ratio_height)
        target['bboxes'] = target['bboxes'] * torch.tensor([ratio_width, ratio_height] * 2)
        # # target['seg_pts'] = target['seg_pts'] * torch.tensor([ratio_width, ratio_height] * 4)
        # target['seg_pts'] = target['seg_pts'] * torch.tensor([ratio_width, ratio_height])

        return rescaled_image, target, rescaled_gt

    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)


class RandomDistortion(object):
    def __init__(self, brightness, contrast, saturation, hue, prob):
        self.prob = prob 
        self.tfm = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target, gt):
        if random.random() > self.prob:
            return image, target, gt 
        return self.tfm(image), target, gt


class ToTensor(object):
    def __call__(self, image, target, gts):
        return F.to_tensor(image), target, gts


class Normalize(object):

    def __init__(self):
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


    def __call__(self, image, target, gt):
        if target is None:
            return image, target, gt 
        
        image = self.normalize(image)
        # h, w = image.shape[-2:]
        # # target['bboxes'] = target['bboxes'] / torch.tensor([w, h] * 2, dtype=torch.float32)
        # # target['seg_pts'] = target['seg_pts'] / torch.tensor([w, h] * 4, dtype=torch.float32)
        # target['seg_pts'] = target['seg_pts'] / torch.tensor([w, h], dtype=torch.float32)
        return image, target, gt


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, gts):
        for t in self.transforms:
            image, target, gts = t(image, target, gts)
        return image, target, gts

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string