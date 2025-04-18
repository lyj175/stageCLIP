import os
import random
import sys

from PIL import Image
import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution), 
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)

def check_sample_integrity(sample_path,type):
    required_files = ['0.'+type, '1.'+type, '2.'+type, '3.'+type]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(sample_path, file)):
            missing_files.append(file)
    return len(missing_files) == 0, missing_files

class MDDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.size = opt["patch_size"]
        self.deg_types = opt["distortion"]

        self.distortion = {}
        for deg_type in opt["distortion"]:
            # 获取所有样本文件夹
            sample_folders = []
            all_folders = sorted([f for f in os.listdir(os.path.join(opt["dataroot"], deg_type)) 
                                if os.path.isdir(os.path.join(opt["dataroot"], deg_type, f))])
            # img_type = os.listdir(os.path.join(opt["dataroot"], deg_type,'1'))[0].split('.')[-1]
            img_type = os.listdir((os.path.join(opt["dataroot"], deg_type)+'/'+os.listdir(os.path.join(opt["dataroot"], deg_type))[0]))[0].split('.')[-1]
            self.img_type = img_type
            for folder in all_folders:
                folder_path = os.path.join(opt["dataroot"], deg_type, folder)
                is_complete, missing = check_sample_integrity(folder_path,self.img_type)
                if is_complete:
                    sample_folders.append(folder)
                else:
                    print(f"警告：样本 {folder_path} 不完整，缺少文件：{missing}")
                
            self.distortion[deg_type] = sample_folders
        self.data_lens = [len(self.distortion[deg_type]) for deg_type in self.deg_types]

        self.random_scale_list = [1]

    def __getitem__(self, index):
        type_id = int(index % len(self.deg_types))
        if self.opt["phase"] == "train":
            deg_type = self.deg_types[type_id]
            index = np.random.randint(self.data_lens[type_id])
        else:
            while index // len(self.deg_types) >= self.data_lens[type_id]:
                index += 1
                type_id = int(index % len(self.deg_types))
            deg_type = self.deg_types[type_id]
            index = index // len(self.deg_types)

        # 获取样本文件夹路径
        sample_folder = self.distortion[deg_type][index]
        sample_path = os.path.join(self.opt["dataroot"], deg_type, sample_folder)

        # 读取GT图片
        self.img_type = os.listdir(sample_path)[0].split('.')[-1]
        GT_path = os.path.join(sample_path, "0."+self.img_type)
        img_GT = util.read_img(None, GT_path, None)  # return: Numpy float32, HWC, BGR, [0,1]

        # 读取所有LQ图片
        img_LQ_list = []
        for i in range(1, 4):  # 读取1.png, 2.png, 3.png
            LQ_path = os.path.join(sample_path, f"{i}."+self.img_type)
            img_LQ = util.read_img(None, LQ_path, None)
            img_LQ_list.append(img_LQ)

        if self.opt["phase"] == "train":
            H, W, C = img_GT.shape

            rnd_h = random.randint(0, max(0, H - self.size))
            rnd_w = random.randint(0, max(0, W - self.size))
            img_GT = img_GT[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :]
            
            # 对所有LQ图片进行相同的裁剪
            img_LQ_list = [img[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :] 
                          for img in img_LQ_list]

            # augmentation - flip, rotate
            # img_LQ_list, img_GT = util.augment(
            #     img_LQ_list + [img_GT],
            #     self.opt["use_flip"],
            #     self.opt["use_rot"],
            #     mode=self.opt["mode"],
            # )
            img_LQ_list = util.augment(
                img_LQ_list + [img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                mode=self.opt["mode"],
            )
            img_GT = img_LQ_list.pop()  # 最后一个元素是GT

        # change color space if necessary
        if self.opt["color"]:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]
            img_LQ_list = [util.channel_convert(img.shape[2], self.opt["color"], [img])[0] 
                          for img in img_LQ_list]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ_list = [img[:, :, [2, 1, 0]] for img in img_LQ_list]

        #TODO 随机选择一张LQ图片用于CLIP
        # lq4clip = []
        # for img in img_LQ_list:
        #     lq4clip.append(clip_transform(img))
        lq4clip = clip_transform(random.choice(img_LQ_list))

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ_list = [torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() 
                      for img in img_LQ_list]

        return {
            "GT": img_GT, 
            "LQ": img_LQ_list,  # 现在返回LQ图片列表
            "LQ_clip": lq4clip,
            "type": deg_type, 
            "GT_path": GT_path
        }

    def __len__(self):
        return np.sum(self.data_lens)

