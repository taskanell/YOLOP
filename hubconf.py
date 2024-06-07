# YOLOP by hustvl, MIT License
dependencies = ['torch']
import torch
import time
from lib.utils.utils import select_device
from lib.config import cfg
from lib.models import get_net
from pathlib import Path
import sys
sys.path.append('/home/iccs/git/object_detection/yolop/tools')
from yolop_detect import detect
import os

def yolop(pretrained=True, device="cpu",image=None, mod=None, mod2=None,conf_thres=0.5, iou_thres=0.45, imshow_title='YOLO',draw_bb_line=True):
    """Creates YOLOP model
    Arguments:
        pretrained (bool): load pretrained weights into the model
        wieghts (int): the url of pretrained weights
        device (str): cuda device i.e. 0 or 0,1,2,3 or cpu
    Returns:
        YOLOP pytorch model
    """
    device = select_device(device = device)
    if image is None or mod is None:
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
            #t0 = time.time()
            res = detect(model=mod,model_y5=mod2,device=device,img=image,conf_thres=conf_thres,iou_thres=iou_thres,imshow_title=imshow_title,draw_bb_line=draw_bb_line)
            #print('HUB TIME', time.time()-t0)
            return res

#def hub_detect(model,device):
    
    #detect(cfg,opt=opt)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='', help='image name')
    opt = parser.parse_args()


