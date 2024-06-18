import cv2
import torch

from depth_anything.dpt import DepthAnything

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))

# # take depth-anything-v2-large as an example
# model = DepthAnything(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
# model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
# model.eval()

# raw_img = cv2.imread('/home/wangru/Project/dog.jpg')
# depth = model.infer_image(raw_img) # HxW raw depth map
