from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def save_img(tensors):
    for i in range(0,len(tensors[0])):
        pil_image = transforms.ToPILImage()(tensors[i])
        plt.imshow(pil_image)
        plt.savefig('my_image.png')
        plt.savefig(f'/home/lee/PycharmProjects/stageCLIP/demo_img/my_image_{i}.png')