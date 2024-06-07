## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import time

def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))
    
def is_between_lines(x, y, line1_params, line2_params):
    #print(len(line2_params))
    if len(line2_params):
        m1, c1 = line1_params
        m2, c2 = line2_params

        #y1 = m1 * x + c1
        #y2 = m2 * x + c2

        x1 = (y - c1)/m1
        x2 = (y - c2)/m2
        
        x1_array = np.array(x1)
        x2_array = np.array(x2)

        #return min(x1, x2) <= x <= max(x1, x2)
        return np.logical_and(x1_array <= x, x <= x2_array)
    else: 
        return False
'''        
def for_function(x,y,line1_params,line2_params):
    if len(line2_params):
        m1, c1 = line1_params
        m2, c2 = line2_params
    
        x1 = (y - c1)/m1
        x2 = (y - c2)/m2
	
        return min(x1, x2) <= x <= max(x1, x2)
    else:
        return False
'''

def show_seg_result(img, result, index, epoch, pixel_mask, f_lines, sec_lines, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False):
    # img = mmcv.imread(img)
    #image = np.copy(img)
    # seg = result[0]
    '''
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
    '''
    color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    left_mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    right_mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
        
    idxs = np.where(pixel_mask==1)
    wan_ids = np.unique(idxs[0])
        
        #print(idxs)
    '''    
    t00 = time.time()
    for y in wan_ids:
       for x in range(img.shape[1]):
            if for_function(x,y,tuple(f_lines[0]),tuple(sec_lines[0])):
                left_mask[y,x] = 1
            if for_function(x,y,tuple(f_lines[1]),tuple(sec_lines[1])):
                right_mask[y,x] = 1
    print('FOR LOOP exec time: %3fs' % (time.time()-t00))
    '''
        #INSTEAD OF FOR / TOO SLOW
        
    x_values = np.arange(img.shape[1])
    y_values = np.arange(img.shape[0])
        #print(y_values.shape)
	
    if len(wan_ids):
       #print('True')
       if len(sec_lines[0]):
          #t0 = time.time()
          left_mask = is_between_lines(x_values, y_values[:, np.newaxis], tuple(sec_lines[0]),tuple(f_lines[0]))
          left_mask [:wan_ids[0],:] = 0
          left_mask [wan_ids[-1]:,:] = 0
          #t_f = time.time()-t0
       if len(sec_lines[1]):
          #t01 = time.time()
          right_mask = is_between_lines(x_values, y_values[:, np.newaxis], tuple(f_lines[1]), tuple(sec_lines[1]))
          right_mask [:wan_ids[0],:] = 0
          right_mask [wan_ids[-1]:,:] = 0
          #print('ALTERNATIVE SOLUTION exec time: %3fs' % (time.time()-t01+t_f))  

        #print(np.count_nonzero(left_mask==1))
        #print(right_mask)
                   
        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

       color_area[result[0] & left_mask == 1] = [0, 255, 0]
       color_area[result[0] & right_mask == 1] = [0, 0, 255]
        #color_area[result[0]==1] = [0, 255, 0]
        
        
       all_right = np.count_nonzero(right_mask == 1)
       all_left = np.count_nonzero(left_mask == 1)
        
       if all_right != 0:
          right_lane_rate = np.count_nonzero(color_area[right_mask == 1]!=[0, 0, 255])/all_right
          right_lane_condition = right_lane_rate < 0.45
       else:
          right_lane_condition = False
           
       if all_left != 0:
          left_lane_rate = np.count_nonzero(color_area[left_mask == 1]!=[0, 255, 0])/all_left
          left_lane_condition = left_lane_rate < 0.45
       else:
          left_lane_condition = False
    else:   
       left_lane_condition = False
       right_lane_condition = False
        #color_area[y_ar[(len(y_ar)-1)//2:],x_ar[(len(y_ar)-1)//2:],:] = [0,0,155]
        
       
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    #img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)
    '''
    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    '''
    return img,left_lane_condition,right_lane_condition
    #return img,False, False


def check_box_lane_localization(x,img,cur_lane_lines=(),draw_line=True):
	
   c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
   width = int(x[2]-x[0])
    
   w = img.shape[1]
   h = img.shape[0]
    
   p1 = int(x[0]+width/2)
   
   if draw_line:
       cv2.line(img,(w//2,h),(p1,int(x[3])),color=(0,0,255),thickness=2)
	
   if len(cur_lane_lines):
       if is_between_lines(p1,int(x[3]),tuple(cur_lane_lines[0]),tuple(cur_lane_lines[1])):
          #print("IN LANE",p1)
          return True,c1,c2
       else:
          #print("OUT OF LANE",p1)
          return False,c1,c2
   else:
	   return None,c1,c2
	
def plot_one_box(x, img, color=None, label=None, line_thickness=None,cur_lane_lines=(),draw_line=True):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    res = check_box_lane_localization(x,img,cur_lane_lines,draw_line)
    
    cv2.rectangle(img, res[1], res[2], color, thickness=tl, lineType=cv2.LINE_AA)
    
    '''      
    if label:
       tf = max(tl - 1, 1)  # font thickness
       t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
       c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
       cv2.rectangle(img, c1, c2, color, tf, cv2.LINE_AA)  # filled
       #print(label)
       cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    '''
    
    return res[0]

if __name__ == "__main__":
    pass
# def plot():
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#     device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU) if not cfg.DEBUG \
#         else select_device(logger, 'cpu')

#     if args.local_rank != -1:
#         assert torch.cuda.device_count() > args.local_rank
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device('cuda', args.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
#     model = get_net(cfg).to(device)
#     model_file = '/home/zwt/DaChuang/weights/epoch--2.pth'
#     checkpoint = torch.load(model_file)
#     model.load_state_dict(checkpoint['state_dict'])
#     if rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
#     if rank != -1:
#         model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
