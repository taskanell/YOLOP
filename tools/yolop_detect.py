import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#print(BASE_DIR)
sys.path.append(BASE_DIR)

#sys.path.append('D:\myYOLOP\YOLOP')

#print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords, pad_image_to_stride
from lib.utils import plot_one_box,show_seg_result
from lib.utils.plot import check_box_lane_localization
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane, map_coordinates, warp_perspective
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(model,model_y5,device,img,img_size=(800),conf_thres=0.5,iou_thres=0.45,imshow_title='YOLO',draw_bb_line=True):

    #logger, _, _ = create_logger(
        #cfg, cfg.LOG_DIR, 'demo')
    
    img = np.ascontiguousarray(img)
    img = pad_image_to_stride(img, stride=32, warn=False)
    #print('Image shape:',img.shape)
    #device = select_device(logger,opt.device)
    #if os.path.exists(opt.save_dir):  # output dir
        #shutil.rmtree(opt.save_dir)  # delete dir
    #os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #model = get_net(cfg)
    #checkpoint = torch.load(opt.weights, map_location= device)
    #model.load_state_dict(checkpoint['state_dict'])
    #model = model.to(device)

    if half:
        model.half()  # to FP16
        if model_y5:
           model_y5.half()
           
    if model_y5:
        y5_image = np.copy(img)
        names = model_y5.module.names if hasattr(model_y5, 'module') else model_y5.names
        total_time_y5 = AverageMeter()
    else:
        total_time_y5 = 0.0 
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[200,255,200] for _ in range(len(names))]

    # Set Dataloader
    '''
    if source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=img_size)
        bs = 1  # batch_size
        print ('HERE')
    '''
    #img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    
    # Get names and colors
   
    #print('Names',names)
    #print(names)
    ##if needed for yolop
	
    # Run inference
    t0 = time.time()

    #vid_path, vid_writer = None, None
    im = torch.zeros((1, 3, img.shape[0], img.shape[1]), device=device)  # init img
    _ = model(im.half() if half else im) if device.type != 'cpu' else None  # run once
    model.eval()
    
    if model_y5:
        _ = model_y5(im.half() if half else im) if device.type != 'cpu' else None  # run once
        model_y5.eval()
    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    #for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        #print("Image det: {}".format(img_det.shape))
        #print("Shapes: {}".format(shapes[1][1]))
    #source = img
    ###for examining lines
    ###image = img
    img_det = np.copy(img)
    img = transform(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        # Inference
    t1 = time_synchronized()
    det_out, da_seg_out,ll_seg_out= model(img)
    t2 = time_synchronized()
    inf_time.update(t2-t1,img.size(0))
    if model_y5:
        t1_y5 = time_synchronized()
        result = model_y5 (y5_image)
        t2_y5 =time_synchronized()
        total_time_y5.update(t2_y5-t1_y5,img.size(0))
        det = np.array(result.xyxy[0].cpu())
        #print('RESULT:',result.xyxy)
        #det = np.array(result.xyxy.cpu())
        img_det = np.squeeze(result.render())
        #cv2.imshow('hi',img_det)
    else:
           # Apply NMS
        inf_out, _ = det_out
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
        #print(det_pred)       
        t4 = time_synchronized()
        
        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]
    

    #save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

    #_, _, height, width = img.shape
    #h,w,_=img_det.shape
    #pad_w, pad_h = shapes[1][1]
    #pad_w = int(pad_w)
    #pad_h = int(pad_h)
    # print (pad_w,pad_h)
    #ratio = shapes[1][0][1]
    
    ratio = 1
    #tt = time.time() 
    #da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
    da_predict = da_seg_out
    #da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_predict, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        #print(da_seg_mask.shape)
    da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
    #ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
    ll_predict = ll_seg_out
    #print('ll_predict',ll_predict)
    #print(ll_predict.shape)
    #ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_predict, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    # Lane line post-processing
    ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
    #ll_seg_mask, lane_right, lane_left = connect_lane(ll_seg_mask)
    #print(len(connect_lane(ll_seg_mask)))
    #if len(connect_lane(ll_seg_mask)) == 4:
       #ll_seg_mask, right_lane, left_lane, lines = connect_lane(ll_seg_mask)
    #if len(connect_lane(ll_seg_mask)) == 3:
    t11 = time.time()
    lane_result = connect_lane(ll_seg_mask)
    #print("DA AND LL",time.time()-tt)
    print("CONNECT_LANE",time.time()-t11)
    if len(lane_result) == 3:
        right_lane, left_lane, lines = lane_result
    else:
        left_lane = []
        right_lane = []
    
    #print(len(lines))
    
    cur_lane_lines = ()
    '''
    if len(lines):
        
        for line in lines:
            coords = map_coordinates(image,np.array(line))
            for x1, y1, x2, y2 in coords:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                cv2.imshow('Lines segmentation ',image)
                cv2.waitKey(1)
    '''
    tt = time.time()
    if len(left_lane) and len(right_lane):
        #print(np.array(right_lane))
        #print(np.array(left_lane))
        #print(image.shape)
        
        
        cur_lane_lines = (left_lane, right_lane)
        
        lines = [line for line in lines if (line[0] !=left_lane[0] and line[0]!= right_lane[0])]
        #print('f_lines',lines)
        #for line in lines: 
            #print(abs(line[0]-left_lane[0]))
            #print(abs(line[0]-right_lane[0]))
			
			
        lines = [line for line in lines if (abs(line[0]-left_lane[0]) > 0.1 and abs(line[0]-right_lane[0]) > 0.1)]
        #print('s_lines',lines)
        #print (lines)
        #if len(lines) / 2 >= 1:
        
        a_array = np.array([ar[0] for ar in lines])
        if len(a_array):
        #print('a_first',a_array)
        ### TOO SMALL a difference
        #for i,a in enumerate(a_array):
           #if abs(a-left_lane[0]) < 0.01 or abs(a-right_lane[0])< 0.01:
               #print('TRUE')
               #a_array= np.delete(a_array, i)
        #print('a',a_array)
            left_idx = np.argmin(a_array)
            right_idx = np.argmax(a_array)
        #print('r',right_idx)
        #print(left_idx==right_idx)
            if lines[left_idx][0] < 0:
               sec_left_line = lines[left_idx]
            else:
               sec_left_line = [] 
            if lines[right_idx][0] > 0:
               sec_right_line = lines[right_idx]
            else:
               sec_right_line = []
        else:
            sec_right_line = []
            sec_left_line = []
            #print(a_array)
        #print(sec_left_line)
        #print('sr',sec_right_line)
        ###for examining lines
        
        right_line_coords = map_coordinates(img_det,np.array(right_lane))

        #x_point = (right_lane[1] - left_lane[1]) / (left_lane[0] - right_lane[0])
        #y_point =  left_lane[0] * x_point + left_lane[1]
        #cv2.circle(image, (int(x_point), int(y_point)), 10, (100, 100, 100), 10)

        #print(x_point,y_point)

        #right_lane_coords = map_coordinates(image,np.array([-0.82, 506.27]))
        left_line_coords = map_coordinates(img_det,np.array(left_lane))
        #print(right_line_coords)
        #print(left_line_coords)
        
        x1_ll , y1 , x2_ll, y2 = left_line_coords[0]
        x1_rl , y1 , x2_rl, y2 = right_line_coords[0]
        #print(x1_ll,x2_ll)

        low_mid = (x1_ll + x1_rl) / 2
        #print(low_mid)
        #up_mid = (x2_ll + x2_rl) / 2
		
        
#here   
        ##uncomment to see the center lane prediction
        ##cv2.circle(img_det, (int(low_mid), y1), 10, (0, 0, 100), 10)
        
        #cv2.circle(image, (int(up_mid), y2), 10, (0, 0, 100), 10)
        

        # Unpack line by coordinates
            
        for x1, y1, x2, y2 in right_line_coords:
            # Draw the line on the created mask 
            try:
               cv2.line(img_det, (x1, y1), (x2, y2), (255, 255, 0), 5)
            except cv2.error as e:
               pass
            #cv2.circle(image, (x1, y1), 10, (0, 0, 100), 10)
            #cv2.circle(image, (x2, y2), 10, (0, 100, 0), 10)
            
            
        for x1, y1, x2, y2 in left_line_coords:
            # Draw the line on the created mask 
            #cv2.circle(image, (x1, y1), 10, (0, 55, 0), 10)
            #cv2.circle(image, (x2, y2), 10, (0, 55, 0), 10)
            try:
               cv2.line(img_det, (x1, y1), (x2, y2), (255, 255, 0), 5)
            except cv2.error as e:
               pass
         
         
        #FIX THAT, UNESSESARY BOTH OF THEM
        pixel_mask = np.all(img_det == (255,255,0), axis =-1)
        #right_pixel_mask = np.all(img_det == (0,255,0), axis =-1)
        pixel_mask = pixel_mask.astype(np.uint8)
        
        #print(pixel_mask)
        
        #right_pixel_mask = right_pixel_mask.astype(np.uint8)
        
        #cv2.imshow('Lane Lines segmentation ',image)
        #cv2.waitKey(1)
        
        #skyview_img = warp_perspective(image)
        #cv2.imshow('exp',skyview_img)
        #cv2.waitKey(0)

        #print(np.unique(ll_seg_mask))

        img_det, turn_left_lane, turn_right_lane = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True,pixel_mask=pixel_mask,f_lines=(left_lane,right_lane),sec_lines=(sec_left_line,sec_right_line))
        #print(len(det))
        #cv2.imshow("img_det",img_det)
        #cv2.waitKey(1)
        #print(img_det[480,640])
        if turn_left_lane:
            print('YOU CAN ENTER LEFT LANE')
            print('\n\n')
        if turn_right_lane:
            print('YOU CAN ENTER RIGHT LANE')
            print('\n\n')
    else:
		
        turn_left_lane = False
        turn_right_lane = False
        print("LANES NOT FOUND")
			
		
    
    bboxes = np.empty(shape=[0,4])
    bboxes_in_lane = []


    
    if model_y5:
        if det.shape[0]!=0:	
            bboxes = det
            print("BBOXES: ",bboxes)
            #for *xyxy,conf,cls in det[0]:
            for *xyxy,conf,cls in det:
			    #print(xyxy)
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                #print('label',label_det_pred)
                bool_value = check_box_lane_localization(xyxy, img_det ,cur_lane_lines=cur_lane_lines, draw_line = draw_bb_line)
                bboxes_in_lane.append(bool_value[0])
    
    else:
        if len(det):
			 #det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
             #print(det.shape[0])
             bboxes = reversed(det[:,:4]).cpu().numpy()
             for *xyxy,conf,cls in reversed(det):
                 label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                 #print('label',label_det_pred)
                 bool_value = plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2, cur_lane_lines=cur_lane_lines, draw_line=draw_bb_line)
                 bboxes_in_lane.append(bool_value)
    
    print('PROCESS TIME:',time.time()-tt)  
    
        #if dataset.mode == 'images':
            #cv2.imwrite(save_path,img_det)

        #elif dataset.mode == 'video':
            #if vid_path != save_path:  # new video
                #vid_path = save_path
                #if isinstance(vid_writer, cv2.VideoWriter):
                    #vid_writer.release()  # release previous video writer

                #fourcc = 'mp4v'  # output video codec
                #fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #h,w,_=img_det.shape
                #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #vid_writer.write(img_det)
        
        #else:
            #cv2.imshow('image', img_det)
            #cv2.waitKey(1)  # 1 millisecond

    #print('Results saved to %s' % Path(opt.save_dir))
    total_time_y5 = total_time_y5.avg if total_time_y5 !=0.0 else 0.0
    print('Done. (%.3fs)' % (time.time() - t0))
    print('yolov5 total avg time (%.3fs/frame)' % (total_time_y5))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    #print(bboxes)
    img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
    return img_det, bboxes, bboxes_in_lane, turn_left_lane, turn_right_lane




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
