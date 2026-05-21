# train
NCCL_DEBUG=INFO  NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2 python -W ignore -m torch.distributed.launch --nproc_per_node=3 --master_port=25641 "/data/***/code/DSA_temporal/tools_DDP/train.py" --dist True -net sam -mod segtask -exp_name dsa/trai_log -sam_ckpt "path_of_sam_cept" -b 1 -image_size 800 -chunk 3 -DVP_mode Noprompt --num_workers 8 

