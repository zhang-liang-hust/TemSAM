
import sys
import torch
from torch.autograd import Function
import logging
import os
import time
from datetime import datetime
import dateutil.tz
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel
from random import sample
import random


#single frame
def get_file_paths_dias(root,split_dir,mode='training'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')#split.txt
    image_dir=os.path.join(root,mode,'images')
    labels_dir=os.path.join(root,mode,'labels')
    image_list = open(split_file, 'r').read().splitlines()#list for images
    allims_list=os.listdir(image_dir)
    for file in image_list:
        sub_files = [filename for filename in allims_list if 'image_s'+str(file) in filename]#name
        sub_files.sort(key=lambda x: int(x.split('_i')[1].split('.')[0])) 
        mask_dir=os.path.join(labels_dir,'label_s'+str(file)+'.png')
        # case_list = []
        for i, sub_file in enumerate(sub_files):
            # case_list.append(os.path.join(image_dir,sub_file))
            img_paths.append(os.path.join(image_dir,sub_file))
            mask_paths.append(mask_dir)
        print('img:', img_paths[-1])
        print('mask:', mask_paths[-1])
    return img_paths,mask_paths

def get_file_paths_dias_case(root,split_dir,mode='training',chunk=5):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')#split.txt
    image_dir=os.path.join(root,mode,'images')
    labels_dir=os.path.join(root,mode,'labels')
    image_list = open(split_file, 'r').read().splitlines()#list for images
    allims_list=os.listdir(image_dir)
    for file in image_list:
        sub_files = [filename for filename in allims_list if 'image_s'+str(file) in filename]#name
        sub_files.sort(key=lambda x: int(x.split('_i')[1].split('.')[0])) 
        mask_dir=os.path.join(labels_dir,'label_s'+str(file)+'.png')
        case_list = []
        for i, sub_file in enumerate(sub_files):
            case_list.append(os.path.join(image_dir,sub_file))
        img_paths.append(case_list)
        mask_paths.append(mask_dir)
        print('img:', img_paths[-1])
        print('mask:', mask_paths[-1])
    return img_paths,mask_paths

def get_file_paths_dias_5frames(root,split_dir,mode='training',chunk=5):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')#split.txt
    image_dir=os.path.join(root,mode,'images')
    labels_dir=os.path.join(root,mode,'labels')
    image_list = open(split_file, 'r').read().splitlines()#list for images
    allims_list=os.listdir(image_dir)
    for file in image_list:
        sub_files = [filename for filename in allims_list if 'image_s'+str(file)+'_i' in filename]#name
        sub_files.sort(key=lambda x: int(x.split('_i')[1].split('.')[0])) 
        mask_dir=os.path.join(labels_dir,'label_s'+str(file)+'.png')
        if(len(sub_files)<=chunk):
            case_list = []
            for i, sub_file in enumerate(sub_files):
                case_list.append(os.path.join(image_dir,sub_file))
            for i  in range((chunk-len(sub_files))):
                case_list.append(os.path.join(image_dir,sub_file))#持续插入最末尾的帧
            img_paths.append(case_list)
            mask_paths.append(mask_dir)
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])
            print("\n")
        else:
            for i in range((len(sub_files)-chunk+1)):#滑窗次数
                sub_list = []
                for k in range(chunk):#窗口大小
                    sub_list.append(os.path.join(image_dir,sub_files[k+i]))
                img_paths.append(sub_list)
                mask_paths.append(mask_dir)
                print('img:', img_paths[-1])
                print('mask:', mask_paths[-1])
                print("\n")
    return img_paths,mask_paths

def get_file_paths_dias_3frames_sliding(root,split_dir,mode='training',chunk=3):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')#split.txt
    image_dir=os.path.join(root,mode,'images')
    labels_dir=os.path.join(root,mode,'labels')
    image_list = open(split_file, 'r').read().splitlines()#list for images
    allims_list=os.listdir(image_dir)
    for file in image_list:
        sub_files = [filename for filename in allims_list if 'image_s'+str(file)+'_i' in filename]#name
        sub_files.sort(key=lambda x: int(x.split('_i')[1].split('.')[0])) 
        mask_dir=os.path.join(labels_dir,'label_s'+str(file)+'.png')

        for i, sub_file in enumerate(sub_files):
            sub_list = []
            adj = 1 #2,3
            #frame1
            if (i-adj) >= 0:
                sub_list.append(os.path.join(image_dir ,sub_files[i-adj]))
            else:
                sub_list.append(os.path.join(image_dir ,sub_file))
            #frame2
            sub_list.append(os.path.join(image_dir ,sub_file))
            #frame3
            if (i+adj) < len(sub_files):
                sub_list.append(os.path.join(image_dir ,sub_files[i+adj]))
            else:
                sub_list.append(os.path.join(image_dir, sub_file))
            img_paths.append(sub_list)
            mask_paths.append(mask_dir)
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])

    return img_paths,mask_paths


# selected three high-quality frames
def get_file_paths_selectedframes(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    for file in image_list:
        img_paths.append(os.path.join(img_dir, file, 'frame_3.png'))
        mask_paths.append(os.path.join(mask_dir, file+'.png'))
    return img_paths, mask_paths

def get_case_paths(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    for file in image_list:
        sub_dir = os.path.join(img_dir, mode, file)
        sub_files = os.listdir(sub_dir)
        sub_files.sort(key=lambda x:int(x.split('.')[0]))
        # if len(sub_files)<5:
        #     for i in range(5-len(sub_files)):
        #         sub_files.insert(0, sub_files[0]) 
        case_list = []
        for i, sub_file in enumerate(sub_files):
            case_list.append(os.path.join(img_dir, mode, file, sub_file))
        img_paths.append(case_list)
        mask_paths.append(os.path.join(mask_dir, file+'.png'))
        
        print('img:', img_paths[-1])
        print('mask:', mask_paths[-1])
    return img_paths, mask_paths

def get_file_paths(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    for file in image_list:
        sub_dir = os.path.join(img_dir, mode, file)
        sub_files = os.listdir(sub_dir)
        sub_files.sort(key=lambda x:int(x.split('.')[0])) 
        for i, sub_file in enumerate(sub_files):
            img_paths.append(os.path.join(img_dir, mode, file, sub_file))
            mask_paths.append(os.path.join(mask_dir, file+'.png'))
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])
    return img_paths, mask_paths
def get_file_paths_singleframe(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    for file in image_list:
        cls=os.listdir(os.path.join(mask_dir,file))#numder of class
        for i in cls:
            sub_dir = os.path.join(mask_dir, file, i)
            mask_paths.append(sub_dir)
            img_paths.append(os.path.join(img_dir, file+'.png'))
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])
    return img_paths, mask_paths
def get_file_paths_singleframe_agument(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    image_list.sort()
    imgpath_list=os.listdir(img_dir)
    for file in image_list:

        sub_dir = os.path.join(mask_dir, file+'.png')
        for sub_img in imgpath_list:
            if str(file) in sub_img:
                mask_paths.append(sub_dir)
                img_paths.append(os.path.join(img_dir,sub_img))

        print('img:', img_paths[-1])
        print('mask:', mask_paths[-1])
    return img_paths, mask_paths
def get_file_paths_singleframe_arcade_rca(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    image_list.sort()
    imgpath_list=os.listdir(img_dir)
    for file in image_list:#img
        sub_dir = os.path.join(mask_dir, file)
        for sub_mask in os.listdir(sub_dir):
            mask_paths.append(os.path.join(mask_dir,file,sub_mask))
            img_paths.append(os.path.join(img_dir,file+'.png'))

        print('img:', img_paths[-1])
        print('mask:', mask_paths[-1])
    return img_paths, mask_paths

# # long frames: three frames
def get_file_paths_sliding_window3(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    for file in image_list:
        sub_dir = os.path.join(img_dir, mode, file)
        sub_files = os.listdir(sub_dir)
        sub_files.sort(key=lambda x:int(x.split('.')[0])) 
        #here sample to 8 frames

        num_files = len(sub_files)

        if num_files > 8:
            step = num_files / 8  
            selected_indices = [round(i * step) for i in range(8)] 
            selected_indices = sorted(set(selected_indices))  
            while len(selected_indices) < 8:  
                extra_idx = random.choice([i for i in range(num_files) if i not in selected_indices])
                selected_indices.append(extra_idx)
                selected_indices.sort()  
            sub_files = [sub_files[i] for i in selected_indices]

        for i, sub_file in enumerate(sub_files):
            
            sub_list = []
            adj = 1 #2,3
            #frame1
            if i-adj<0 or i+adj>=len(sub_files):
                continue
            if (i-adj) >= 0:
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i-adj]))
            else:
                sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            #frame2
            sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            #frame3
            if (i+adj) < len(sub_files):
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i+adj]))
            else:
                sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            img_paths.append(sub_list)
            mask_paths.append(os.path.join(mask_dir, file+'.png'))
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])
    return img_paths, mask_paths

def get_file_paths_dias(img_dir, mask_dir, mode='train'):
    img_paths = []
    mask_paths = []

    image_list = os.listdir(img_dir)
    for file in image_list:
        sub_dir = os.path.join(img_dir, file)
        sub_files = os.listdir(sub_dir)# subframes
        sub_files.sort(key=lambda x:int(x.split('.')[0])) 
        #here sample to 8 frames


        for i, sub_file in enumerate(sub_files):
            sub_path=os.path.join(img_dir,file,sub_file)
            img_paths.append(sub_path)
            mask_paths.append(os.path.join(mask_dir, file+'.png'))
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])
    return img_paths, mask_paths
    
def get_file_paths_sliding_window3_dsca(img_dir, mask_dir, mode='train'):
    img_paths = []
    mask_paths = []

    image_list = os.listdir(img_dir)
    for file in image_list:
        sub_dir = os.path.join(img_dir, file)
        sub_files = os.listdir(sub_dir)# subframes
        sub_files.sort(key=lambda x:int(x.split('.')[0])) 
        #here sample to 8 frames

        num_files = len(sub_files)
        if num_files > 8:
            step = num_files / 8  
            selected_indices = [round(i * step) for i in range(8)] 
            selected_indices = sorted(set(selected_indices)) 
            while len(selected_indices) < 8: 
                extra_idx = random.choice([i for i in range(num_files) if i not in selected_indices])
                selected_indices.append(extra_idx)
                selected_indices.sort() 
            sub_files = [sub_files[i] for i in selected_indices]

        for i, sub_file in enumerate(sub_files):
            sub_list = []
            adj = 1 #2,3
            #frame1
            if i-adj<0 or i+adj>=len(sub_files):
                continue
            if (i-adj) >= 0:
                sub_list.append(os.path.join(img_dir, file, sub_files[i-adj]))
            else:
                sub_list.append(os.path.join(img_dir, file, sub_file))
            #frame2
            sub_list.append(os.path.join(img_dir, file, sub_file))
            #frame3
            if (i+adj) < len(sub_files):
                sub_list.append(os.path.join(img_dir, file, sub_files[i+adj]))
            else:
                sub_list.append(os.path.join(img_dir, file, sub_file))
            img_paths.append(sub_list)
            mask_paths.append(os.path.join(mask_dir, file+'.png'))
            print('img:', img_paths[-1])
            print('mask:', mask_paths[-1])
    return img_paths, mask_paths


# long frames: five frames
def get_file_paths_sliding_window5(img_dir, mask_dir, split_dir, mode='train'):
    img_paths = []
    mask_paths = []
    split_file = os.path.join(split_dir, mode+'.txt')
    image_list = open(split_file, 'r').read().splitlines()
    for file in image_list:
        sub_dir = os.path.join(img_dir, mode, file)
        sub_files = os.listdir(sub_dir)
        sub_files.sort(key=lambda x:int(x.split('.')[0])) 
        for i, sub_file in enumerate(sub_files):
            sub_list = []
            adj = 1
            # adj = 2 
            # adj = 3
            #frame1
            if (i-2*adj) >= 0:
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i-2*adj]))
            elif (i-adj) >= 0:
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i-adj]))
            else:
                sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            #frame2
            if (i-adj) >= 0:
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i-adj]))
            else:
                sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            #frame3
            sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            #frame4
            if (i+adj) < len(sub_files):
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i+adj]))
            else:
                sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            #frame5
            if (i+2*adj) < len(sub_files):
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i+2*adj]))
            elif (i+adj) < len(sub_files):
                sub_list.append(os.path.join(img_dir, mode, file, sub_files[i+adj]))
            else:
                sub_list.append(os.path.join(img_dir, mode, file, sub_file))
            img_paths.append(sub_list)
            mask_paths.append(os.path.join(mask_dir, file+'.png'))
            # print('img:', img_paths[-1])
            # print('mask:', mask_paths[-1])
    return img_paths, mask_paths


def get_network_sequence_hqsam_patch800_mip_wholsequence_maskattention_decoder(args, net, use_gpu=True, gpu_device = 0, distribution = True, multi_task=False):
    """ return given network
    """
    if net == 'sam':
        if multi_task: 
            from models.hqsam_sequence_v2_patch800_mip_wholesequence_maskattention_decoder import SamPredictor, sam_model_registry_multitask
            from models.hqsam_sequence_v2_patch800_mip_wholesequence_maskattention_decoder.utils.transforms import ResizeLongestSide
            net = sam_model_registry_multitask['vit_b'](args,checkpoint=args.sam_ckpt)
        else:
            from models.hqsam_sequence_v2_patch800_mip_wholesequence_maskattention_decoder import SamPredictor, sam_model_registry
            from models.hqsam_sequence_v2_patch800_mip_wholesequence_maskattention_decoder.utils.transforms import ResizeLongestSide
            net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt)
    elif net == 'sam-cnn':
        from models.hqsam_sequence_v2_patch800_mip_wholesequence_maskattention_decoder_cnn import SamPredictor, sam_model_registry
        from models.hqsam_sequence_v2_patch800_mip_wholesequence_maskattention_decoder_cnn.utils.transforms import ResizeLongestSide
        net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt)
        
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    if use_gpu:
        net = net.to(device=gpu_device)
        if args.dist:
            net = DistributedDataParallel(net)
    return net

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)



def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)
