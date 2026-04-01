CUDA_VISIBLE_DEVICES=0,3 torchrun \
--nproc_per_node=2 \
--master_port=29501 \
train.py \
-c conf/imagenet/8_768_300E_t4111_Const_E4.yml \
--model sdt \
--spike-mode lif --experiment imagenet_const_e4_4gpu_lr2e-3