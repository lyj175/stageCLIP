import gradio as gr
import cv2
import argparse
import sys, os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import torchvision.utils as tvutils

import options as option
from models import create_model

sys.path.insert(0, "../../")
import open_clip
import utils as util

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='options/test.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

# load pretrained model by default
model = create_model(opt)#TODO diffusion
device = model.device

#TODO daclip_VIT['daclip'] self trained
# opt['path']['daclip'] = '/home/ligong3/l/daclip-uir/da-clip/src/logs/daclip_ViT-B-32_b784x1_lr2e-5_e50_17/checkpoints/epoch_30.pt'
# opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/da-clip/src/logs/daclip_ViT-B-32-2023-09_b512x1_lr2e-5_e30_test_5/checkpoints/epoch_30.pt'
# opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/daclip_ViT-B-32.pt'
opt['path']['daclip'] = '/home/lee/PycharmProjects/stageCLIP/da-clip/src/logs/2025_04_01-17_02_02-model_daclip_ViT-B-32-lr_2e-05-b_16-j_8-p_amp/checkpoints/epoch_1.pt'
clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
clip_model = clip_model.to(device)

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)


def restore(image):
    image = image / 255.
    img4clip = clip_transform(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_context, degra_context = clip_model.encode_image(img4clip, control=True)
        image_context = image_context.float()
        degra_context = degra_context.float()

    LQ_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    noisy_tensor = sde.noise_state(LQ_tensor)
    #TODO feed text and image context to diffusion to  computing
    model.feed_data(noisy_tensor, LQ_tensor, text_context=degra_context, image_context=image_context)
    model.test(sde)
    visuals = model.get_current_visuals(need_GT=False)
    output = util.tensor2img(visuals["Output"].squeeze())
    return output[:, :, [2, 1, 0]]

examples=[os.path.join(os.path.dirname(__file__), f"images/{i}.jpg") for i in range(1, 20)]
interface = gr.Interface(fn=restore, inputs="image", outputs="image", title="Image Restoration with DA-CLIP", examples=examples)
interface.launch()
