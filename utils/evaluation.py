r""" Evaluate mask prediction """
import torch
import numpy as np

from skimage import morphology
from sklearn.metrics import roc_auc_score
import cv2

def double_threshold_iteration(img, h_thresh, l_thresh):
    h, w = img.shape
    # img = np.array(torch.sigmoid(img).cpu().detach()*255, dtype=np.uint8)
    img= np.array(img*255, dtype=np.uint8)
    bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8)
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h-1):
            for j in range(w-1):
                if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
                    if gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or gbin[i][j-1] or gbin[i][j+1] or gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]:
                        gbin[i][j] = 255
    # 腐蚀操作
    # kernel = np.ones((3, 3), np.uint8)
    # gbin_eroded = cv2.erode(gbin, kernel, iterations=1)
    # gbin=gbin_eroded
    return gbin/255

def computeF1(pred, gt):
    """

    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score * 100, precision * 100, recall * 100

def computeTopo(pred, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: Topo metric
    """
    pred = pred[0].detach().cpu().numpy().astype(int)  # float data does not support bit_and and bit_or
    gt = gt[0].detach().cpu().numpy().astype(int)
    #print(pred.shape)
    pred = morphology.skeletonize(pred >= 0.5)
    gt = morphology.skeletonize(gt >= 0.5)

    cor_intersection = gt & pred

    com_intersection = gt & pred

    cor_tp = np.sum(cor_intersection)
    com_tp = np.sum(com_intersection)

    sk_pred_sum = np.sum(pred)
    sk_gt_sum = np.sum(gt)

    smooth = 1e-7
    correctness = cor_tp / (sk_pred_sum + smooth)
    completeness = com_tp / (sk_gt_sum + smooth)

    quality = cor_tp / (sk_pred_sum + sk_gt_sum - com_tp + smooth)

    return torch.tensor(correctness * 100), torch.tensor(completeness * 100), torch.tensor(quality * 100)
def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    smooth = 1e-7
    # return np.sum(v*s)/np.sum(s)
    return np.sum(v*s)/(np.sum(s)+smooth)

def computeCldice(pred, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: Topo metric
    """
    v_p = pred >= 0.5
    v_l = gt >= 0.5

    tprec = cl_score(v_p, morphology.skeletonize(v_l))
    tsens = cl_score(v_l, morphology.skeletonize(v_p))
    
    smooth = 1e-7
    return torch.tensor(100*2*tprec*tsens/(tprec+tsens+smooth))

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    # if volume_sum == 0:
        # return np.NaN #Ori
    volume_intersect = (mask_gt & mask_pred).sum()
    # return 2*volume_intersect / volume_sum
    smooth = 1e-7
    return 100*(2*volume_intersect / (volume_sum + smooth))

def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result

def get_metrics(predict, target, threshold=0.5):

    
    predict_b = np.where(predict >= threshold, 1, 0)
    cldice = computeCldice(predict_b,target) 
    # cldice = computeCldice(predict_b,target) if run_clDice else 0

    predict = predict.flatten()
    predict_b = predict_b.flatten()
    target = target.flatten()
    if max(target) > 1:
        target = to_one_hot(target, all_seg_labels=[1]).flatten()
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    if np.all(target == 0) or np.all(predict == 0):
        auc = 1
    else:
        auc = roc_auc_score(target, predict)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    DSC = 2 * pre * sen / (pre + sen)
    # return DSC,acc,sen,spe,iou,auc,cldice
    return torch.tensor(np.round(DSC, 4), dtype=torch.float32),\
        torch.tensor(np.round(acc, 4), dtype=torch.float32),\
        torch.tensor(np.round(sen, 4), dtype=torch.float32),\
        torch.tensor(np.round(spe, 4), dtype=torch.float32),\
        torch.tensor(np.round(iou, 4), dtype=torch.float32),\
        torch.tensor(np.round(auc, 4), dtype=torch.float32),\
        torch.tensor(np.round(cldice, 4), dtype=torch.float32)

def count_connect_component(predict, target, connectivity=8):

    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return torch.tensor((pre_n/gt_n),dtype=torch.float32)


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls,pred_mask, gt_mask,threshold=0.5):
        # bs, 1, h, w
        
        DSC = []
        Acc = []
        Sen = []
        Spe = []
        IOU = []
        AUC = []
        cldice = []
        VC = []

        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            DSC_,Acc_,Sen_,Spe_,IOU_,AUC_,cldice_ = get_metrics(_pred_mask,_gt_mask,threshold=threshold)
            VC_ = count_connect_component(np.where(_pred_mask >= threshold, 1, 0), _gt_mask)#binary prob
            DSC.append(DSC_)
            Acc.append(Acc_)
            Sen.append(Sen_)
            Spe.append(Spe_)
            IOU.append(IOU_)
            AUC.append(AUC_)
            cldice.append(cldice_)
            VC.append(VC_)
        DSC=torch.stack(DSC)
        Acc=torch.stack(Acc)
        Sen=torch.stack(Sen)
        Spe=torch.stack(Spe)
        IOU=torch.stack(IOU)
        AUC=torch.stack(AUC)
        cldice=torch.stack(cldice)
        VC=torch.stack(VC)

        return DSC,Acc,Sen,Spe,IOU,AUC,cldice,VC

