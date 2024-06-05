import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import imageio


def save_frame_dvs(x: torch.Tensor or np.ndarray,
                   save_gif_to: str = None) -> None:
    '''
    :param x: frames with ``shape=[T, 2, H, W], in [0,1]``
    :type x: torch.Tensor 
    '''
    x = x.mul(255).add_(0.5).clamp_(0, 255)
    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]
    img_tensor[:, 2] = x[:, 1]
    img_tensor = img_tensor.permute(0, 2, 3, 1).cpu().numpy()
    img_list = []
    for t in range(img_tensor.shape[0]):
        img_list.append(Image.fromarray(img_tensor[t].astype(np.uint8)))
    imageio.mimsave(save_gif_to, img_list, fps=8, loop=0)
    print(f'Save frames to [{save_gif_to}].')
