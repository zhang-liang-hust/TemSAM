
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from conf import settings, cfg
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
)
from utils.evaluation import Evaluator
from utils.logger import AverageMeter
from utils.utils import *
from utils.losses import *

import matplotlib as mpl
mpl.use('Agg')
import cv2
import scipy.ndimage.filters as filters
from models.heat_models.data_utils import collate_fn, get_pixel_features
from models.heat_models.loss import CornerCriterion, Surface_Loss, Sparsity_loss, NVSurface_Loss, DSC_loss
from models.heat_models.metrics.get_metric import compute_corner_metrics, get_recall_and_precision
from models.heat_models.utils.geometry_utils import corner_eval


args = cfg.parse_args()
# GPUdevice = torch.device('cuda', args.gpu_device)
rank = args.gpu_device
if args.dist:
    rank = args.local_rank
    print('rank:',rank)
GPUdevice = torch.device(f"cuda:{rank}")

pos_weight = torch.ones([1]).to(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_G_tmp = torch.nn.BCELoss()

boundary_lsfunc=HausdorffERLoss()
seed = torch.randint(1,11,(args.b,7))
beta=5 # 5 is best here ,we need test other 
lamdba=0.1
torch.backends.cudnn.benchmark = True

image_size = 1024
corner_criterion = CornerCriterion(image_size=image_size)

def DVP_model_forward(imgs, mip, reframe,net):
    if args.dist:
        net = net.module
        
    mip = rearrange(mip, 'b h w c -> b c h w')
    reframe = rearrange(reframe, 'b h w c -> b c h w') # dias need

    imge,mipe,inter_em, refframe_embedding = net.image_encoder(imgs, mip,reframe) 

    if args.DVP_mode=='Noprompt':
        se, de = net.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
    de=mipe # mip as prompt
    pred, _,mask_sam = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se, 
                dense_prompt_embeddings=de, 
                multimask_output=False,
                interm_embeddings=inter_em,
                refframe_embedding=refframe_embedding,
              )
    return pred,mask_sam
    
def train_sam_segtask_liver(args, net: nn.Module, optimizer, train_loader, epoch):
    epoch_loss = 0
    # train mode
    net.train()
    corner_criterion.train()
    optimizer.zero_grad()
    epoch_loss = 0
    epoch_boundary_loss = 0
    epoch_tmp_segloss=0
    epoch_segloss = 0
    lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            seg_labels = pack['seg_labels'].to(dtype = torch.float32, device = GPUdevice)
            mip_img=pack['mip_data'].to(dtype = torch.float32, device = GPUdevice)
            reframe=pack['reframe_data'].to(dtype = torch.float32, device = GPUdevice)#1,800,800,3

            pred,mask_sam= DVP_model_forward(imgs,mip_img,reframe,net)

            mask_sam = F.interpolate(mask_sam,seg_labels.shape[-2:],mode="bilinear", align_corners=False).detach()

            loss_seg = lossfunc(pred, seg_labels) #segmentation loss
            loss_seg_tmp = lossfunc(mask_sam, seg_labels) #segmentation loss

            loss_boundary = boundary_lsfunc(torch.sigmoid(pred), seg_labels)

            loss = loss_seg+beta*loss_boundary+lamdba*loss_seg_tmp

            pbar.set_postfix(**{'seg':loss_seg.item()})
            epoch_loss += loss.item()
            epoch_boundary_loss += (loss_boundary*beta).item()
            epoch_segloss+= (loss_seg).item()
            epoch_tmp_segloss+= (loss_seg_tmp*lamdba).item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del mask_sam
            pbar.update()
    return epoch_loss/len(train_loader),epoch_boundary_loss/len(train_loader),epoch_segloss/len(train_loader),epoch_tmp_segloss/len(train_loader)



def validation_sam_segtask(args, val_loader, epoch, net: nn.Module):
    # eval mode
    net.eval()
    corner_criterion.eval()
    n_val = len(val_loader)  # the number of batch
    mix_res = (0,0,0,0)
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    # threshold = (0.5)

    # GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            seg_labels = pack['seg_labels'].to(dtype = torch.float32, device = GPUdevice)
            mip_img=pack['mip_data'].to(dtype = torch.float32, device = GPUdevice)
            reframe=pack['reframe_data'].to(dtype = torch.float32, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']
            dataset = pack['dataset']

            chunk = imgs.shape[1]
            '''test'''
            with torch.no_grad():
                pred,_ = DVP_model_forward(imgs, mip_img,reframe, net)
    
                loss_seg = lossfunc(pred, seg_labels) #segmentation loss
                loss = loss_seg
                pbar.set_postfix(**{'seg':loss_seg.item()})
                tot += loss
                temp = eval_seg(pred, seg_labels, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
            pbar.update()
    return tot/ n_val , tuple([a/n_val for a in mix_res])
def test_sam_wholeimg_segtask(args, val_loader, epoch, net: nn.Module):
    # eval mode
    net.eval()
    corner_criterion.eval()
    
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    
    Evaluator.initialize()
    average_meter = AverageMeter(val_loader.dataset)
    lossfunc = criterion_G

    all_prec = list()
    all_recall = list()

    corner_tp = 0.0
    corner_fp = 0.0
    corner_length = 0.0
    cldice = 0.0

    # pedciton_dict={}
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)#13,3,1024,1024
            seg_labels = pack['seg_labels'].to(dtype = torch.float32, device = GPUdevice)#1,1,1024,1024
            mip_img=pack['mip_data'].to(dtype = torch.float32, device = GPUdevice)#1,1024,1024,3
            reframe=pack['reframe_data'].to(dtype = torch.float32, device = GPUdevice)#1,1024,1024,3

            dataset = pack['dataset']
            name = pack['image_meta_dict']['filename_or_obj']
            # case_name=name.split('_')[1]
            # frame_id=name.split('_')[2]
            '''test'''
            
            with torch.no_grad():
                keypt_preds,mask_sam = DVP_model_forward(imgs, mip_img,reframe, net)#1,1,1024,1024
                keypt_preds = keypt_preds.sigmoid()#1,1024,1024
                mask_sam = mask_sam.sigmoid()#1,1024,1024

                ###############################################################################

                LOCAL_MAX_THRESH = 0.5 #0.6
                NEIGHBOUR_SIZE = 10
                corner_preds = keypt_preds.detach().cpu().squeeze().numpy()
                preds_sam=mask_sam.detach().cpu().squeeze().numpy()
                data_max = filters.maximum_filter(corner_preds, NEIGHBOUR_SIZE)
                maxima = (corner_preds == data_max)
                data_min = filters.minimum_filter(corner_preds, NEIGHBOUR_SIZE)
                diff = ((data_max - data_min) > 0)
                maxima[diff == 0] = 0
                local_maximas = np.where((maxima > 0) & (corner_preds > LOCAL_MAX_THRESH))
                point_coords = np.stack(local_maximas, axis=-1)[:, [1, 0]]  #xy format!! 

                save_corner_path = os.path.join(args.path_helper['log_path'], dataset[0], 'corner')
                if not os.path.exists(save_corner_path):
                    os.makedirs(save_corner_path)
                cv2.imwrite(os.path.join(save_corner_path, '{}_output.png'.format(name[0])), corner_preds * 255)
                cv2.imwrite(os.path.join(save_corner_path, '{}_samori.png'.format(name[0])), preds_sam * 255)

                binary_output = (corner_preds >= LOCAL_MAX_THRESH).astype(np.int_)
                cv2.imwrite(os.path.join(save_corner_path, '{}_binary_output.png'.format(name[0])), binary_output * 255)

                #visualize pred points
                conf_map = np.zeros_like(corner_preds)
                pred_corners = np.array(point_coords.round())
                xint, yint = pred_corners[:, 0].astype(np.int32), pred_corners[:, 1].astype(np.int32)
                conf_map[yint, xint] = 1
                cv2.imwrite(os.path.join(save_corner_path, '{}_conf.png'.format(name[0])), conf_map * 255)

                viz_image = pack['viz_image'][0].cpu().numpy().transpose(1, 2, 0)
                viz_image = (viz_image * 255).astype(np.uint8)

                pred_mask = torch.where(keypt_preds >= 0.5, 1, 0) #SAM's segment result,1,1,1024,1024
                save_mask = pred_mask.squeeze().cpu().numpy()
                save_path = os.path.join(args.path_helper['log_path'], dataset[0])
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                
                cv2.imwrite(os.path.join(save_path, name[0]+'_prediction.png'), save_mask * 255)
                save_gt = seg_labels.squeeze().cpu().numpy()
                cv2.imwrite(os.path.join(save_path, name[0]+'_gt.png'), save_gt * 255)

                DSC,Acc,Sen,Spe,IOU,AUC,cldice,VC = Evaluator.classify_prediction(keypt_preds.squeeze(1).clone().cpu().numpy(), seg_labels.squeeze(1).cpu().numpy())# prob input
                average_meter.update(DSC,Acc,Sen,Spe,IOU,AUC,cldice,VC)

                recon_path = os.path.join(save_corner_path, '{}_pred_corner.png'.format(name[0]))
                pbar.update()
                
    dsc = average_meter.compute_dsc()
    acc=average_meter.compute_acc()
    sen=average_meter.compute_sen()
    spe=average_meter.compute_spe()
    iou = average_meter.compute_iou()
    auc = average_meter.compute_auc()
    cldice = average_meter.compute_cldice()
    vc = average_meter.compute_vc()
    print('dsc: {:.4f} acc: {:.4f} sen: {:.4f} spe: {:.4f}'.format(dsc.item(), acc.item(), sen.item(), spe.item()))
    print('iou: {:.4f} auc: {:.4f} cldice: {:.4f} vc: {:.4f}'.format(iou.item(), auc.item(), cldice.item(),vc.item()))

    return dsc,acc,sen,spe,iou,auc,cldice,vc

##################################################################
def visualize_cond_generation(positive_pixels, confs, image, save_path, gt_corners=None, prec=None, recall=None, dice=None, cldice=None,
                              image_masks=None, edges=None, edge_confs=None):
    image = image.copy()  # get a new copy of the original image
    if confs is not None:
        viz_confs = confs

    if edges is not None:
        preds = positive_pixels.astype(int)
        c_degrees = dict()
        for edge_i, edge_pair in enumerate(edges):
            conf = (edge_confs[edge_i] * 2) - 1
            cv2.line(image, tuple(preds[edge_pair[0]]), tuple(preds[edge_pair[1]]), (255 * conf, 255 * conf, 0), 2)
            c_degrees[edge_pair[0]] = c_degrees.setdefault(edge_pair[0], 0) + 1
            c_degrees[edge_pair[1]] = c_degrees.setdefault(edge_pair[1], 0) + 1

    for idx, c in enumerate(positive_pixels):
        if edges is not None and idx not in c_degrees:
            continue
        if confs is None:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255), -1)
        else:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 0, 255 * viz_confs[idx]), -1)

    if gt_corners is not None:
        for c in gt_corners:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 0), -1)

    if image_masks is not None:
        mask_ids = np.where(image_masks == 1)[0]
        for mask_id in mask_ids:
            y_idx = mask_id // 64
            x_idx = (mask_id - y_idx * 64)
            x_coord = x_idx * 4
            y_coord = y_idx * 4
            cv2.rectangle(image, (x_coord, y_coord), (x_coord + 3, y_coord + 3), (127, 127, 0), thickness=-1)

    if prec is not None:
        if isinstance(prec, tuple):
            cv2.putText(image, 'edge p={:.2f}, edge r={:.2f}'.format(prec[0], recall[0]), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'region p={:.2f}, region r={:.2f}'.format(prec[1], recall[1]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            if dice is not None:
                cv2.putText(image, 'prec={:.2f}, recall={:.2f}, dice={:.2f}, cldice={:.2f}'.format(prec, recall, dice, cldice), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, 'prec={:.2f}, recall={:.2f}'.format(prec, recall), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)   
    cv2.imwrite(save_path, image)
