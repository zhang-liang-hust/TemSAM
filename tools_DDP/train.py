'''
Dynamic video prompt: Spatio-temporal Prompting Network for Robust Video Feature Extraction(ICCV2023)
'''
import sys
sys.path.append("..")
sys.path.append("/data/***/code/DSA_temporal/")

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset_DSA_keypt import SlidingDataset_DSA_keypt1,SlidingDataset_DSA_keypt3
from data.dataset_dias import *
from data.dataset_dsca import Dataset_DSA_dsca
from conf import settings, cfg
import time
from torch.utils.data import DataLoader
from utils.utils import *
import function as function
# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

args = cfg.parse_args()
rank = args.gpu_device
if args.dist:
    dist.init_process_group(backend="nccl") 
    rank = args.local_rank
    print('rank:',rank)
GPUdevice = torch.device(f"cuda:{rank}")
torch.cuda.set_device(GPUdevice)

root='/data/**/dataset/DIAS/'
split_path=root+'split'
img_paths,mask_paths=get_file_paths_dias_3frames_sliding(root,split_path,mode='training',chunk=args.chunk)
dataset=Dataset_DSA_dias('train',img_paths,mask_paths,size=args.image_size,aug=True,repeat=1)
val_img_paths,val_mask_paths=get_file_paths_dias_3frames_sliding(root,split_path,mode='validation',chunk=args.chunk)
val_dataset=Dataset_DSA_dias('test',val_img_paths,val_mask_paths,size=args.image_size,aug=False,repeat=1)



train_sampler = DistributedSampler(dataset) if args.dist else None
nice_train_loader = DataLoader(dataset, batch_size=args.b, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)

val_sampler = DistributedSampler(val_dataset) if args.dist else None
nice_test_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=args.num_workers, sampler=val_sampler)

###liver dataset
'''create model, optimizer'''
# mode = 'keypttask' # 'segtask','keypttask','multitask'
mode = args.mode
print('mode:',mode)
if mode=='multitask':
    net = get_network_sequence_hqsam_patch800_mip_wholsequence_maskattention_decoder(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed, multi_task=True)
else:
    net = get_network_sequence_hqsam_patch800_mip_wholsequence_maskattention_decoder(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed, multi_task=False)

for n, value in net.module.image_encoder.named_parameters():
    if "Adapter" not in n :
        value.requires_grad = False
    else:
        if rank==0:
            print('n:',n)

for n, value in net.module.prompt_encoder.named_parameters():
    value.requires_grad = False
for n, value in net.module.mask_decoder.named_parameters():
    value.requires_grad = True

total_num = sum(p.numel() for p in net.module.parameters())
trainable_num = sum(p.numel() for p in net.module.parameters() if p.requires_grad)
if rank==0:
    print('total_num:',total_num) #136300336
    print('trainable_num:',trainable_num) #42564864

optimizer = optim.Adam(net.module.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) #learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    net.module.load_state_dict(checkpoint['state_dict'],strict=False)
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    del checkpoint
    torch.cuda.empty_cache()
    
if rank==0:
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''checkpoint path'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
tol = 1e4
edice = 0.0

for epoch in range(settings.EPOCH):
    net.train()
    time_start = time.time()
    if args.dist:
        dist.barrier()
        train_sampler.set_epoch(epoch)
    
    if mode=='segtask':
        loss = function.train_sam_segtask_liver(args, net, optimizer, nice_train_loader, epoch)
    elif mode=='keypttask':
        loss = function.train_sam_keypttask(args, net, optimizer, nice_train_loader, epoch)
    elif mode=='multitask':
        loss = function.train_sam_multitask(args, net, optimizer, nice_train_loader, epoch)
    if rank==0:
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
    time_end = time.time()
    if rank==0:
        print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        if mode=='segtask':
            tol, (eiou, edice) = function.validation_sam_segtask(args, nice_test_loader, epoch, net)
            if rank==0:
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        
    if args.dist:
        sd = net.module.state_dict()
    else:
        sd = net.state_dict()
    if rank==0:
        if(epoch%1==0):
            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, False, args.path_helper['ckpt_path'], filename="epoch"+str(epoch+1)+"_checkpoint")

        if edice >= best_acc:
            best_acc = edice
            is_best_acc = True
            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best_acc, args.path_helper['ckpt_path'], filename="best_dice_checkpoint")
        else:
            is_best_acc = False

        if tol <= best_tol:
            best_tol = tol
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
        else:
            is_best = False
            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename="latest_checkpoint")
    
    scheduler.step()
    if rank==0:
        print('lr', scheduler.get_last_lr())
        