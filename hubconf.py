# YOLOP by hustvl, MIT License
dependencies = ['torch']
import torch
from lib.utils.utils import select_device
from lib.config import cfg
from lib.models import get_net
from pathlib import Path
from tools.yolop_detect import detect
import os

def yolop(pretrained=True, device="cpu",image=None, mod=None, conf_thres=0.5, iou_thres=0.45):
    """Creates YOLOP model
    Arguments:
        pretrained (bool): load pretrained weights into the model
        wieghts (int): the url of pretrained weights
        device (str): cuda device i.e. 0 or 0,1,2,3 or cpu
    Returns:
        YOLOP pytorch model
    """
    device = select_device(device = device)
    if image is None:
        #device = select_device(device = device)
        model = get_net(cfg)
        if pretrained:
           path = os.path.join(Path(__file__).resolve().parent, "weights/End-to-end.pth")
           checkpoint = torch.load(path, map_location= device)
           model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        return model
    else:
        with torch.no_grad():
            detect(model=mod,device=device,img=image,conf_thres=conf_thres,iou_thres=iou_thres)

#def hub_detect(model,device):
    
    #detect(cfg,opt=opt)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='', help='image name')
    opt = parser.parse_args()


