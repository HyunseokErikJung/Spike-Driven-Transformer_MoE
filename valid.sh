# CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29001 firing_num.py -c ./conf/cifar100/2_512_300E_t4.yml -data-dir /dataset/CIFAR100/ --model sdt --spike-mode lif --resume model_best.pth_new.tar --no-resume-opt


# python firing_num.py -c ./conf/cifar100/4_384_300E_t4.yml -data-dir /dataset/CIFAR100/ \
#     --model sdt --spike-mode lif \
#     --resume /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar \
#     --no-resume-opt \
#     --val-batch-size 32


# python visualize_expert_assignment.py \
#   -c conf/cifar100/4_384_300E_t4.yml \
#   --resume /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar \
#   --num-images 16 \
#   --output-dir ./visual --start-idx 3300 --overlay-alpha 0.7


# python eval_routing_masks.py \
#   -c conf/cifar100/4_384_300E_t4.yml \
#   --resume /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar \
#   --val-batch-size 128 \
#   --device cuda:0  


# python analyze_routing_stats.py \
#   -c conf/cifar100/4_384_300E_t4.yml \
#   --resume /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar \
#   --val-batch-size 64 \
#   --device cuda:0 \
#   --output-dir ./visual
  

# python visualize_expert_assignment_bottom1.py \
#   -c conf/cifar100/4_384_300E_t4.yml \
#   --resume /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar \
#   --num-images 16 \
#   --output-dir ./visual --start-idx 3300 --overlay-alpha 0.7

  # python visualize_expert_confidence_overlay.py \
  # -c conf/cifar100/4_384_300E_t4.yml \
  # --resume /home2/hyunseok/Code/26_Spiking_MoE/Spike-Driven-Transformer_MoE_4111expert/output/20260315-000051-sdt-data-cifar100-t-4-spike-lif/model_best.pth.tar \
  # --num-images 8 \
  # --output-dir ./visual --start-idx 3300 --overlay-alpha 0.7

  # CUDA_VISIBLE_DEVICES=7 python visualize_expert_assignment.py \
  # -c conf/cifar100/4_384_300E_t4.yml \
  # --resume ./output/finetune/cifar100_ft_routerKD_KDw0p2-ft-ckpt27bfd333/checkpoint-39.pth.tar \
  # --num-images 16 \
  # --output-dir ./visual_ft --start-idx 3300 --overlay-alpha 0.7


  
CUDA_VISIBLE_DEVICES=7 python visualize_expert_assignment.py \
  -c conf/cifar100/2_512_300E_t41_Const_E2.yml \
    --resume ./output/20260319-2_512_200E_t41_Const_E2/model_best.pth.tar \
    --num-images 4 \
    --output-dir ./visual_T41CE2 --start-idx 3306 --overlay-alpha 0.7

  CUDA_VISIBLE_DEVICES=7 python visualize_expert_assignment.py \
  -c conf/cifar100/2_512_300E_t4111_Const_E4.yml \
    --resume ./output/20260319-2_512_200E_t4111_Const_E4/model_best.pth.tar \
    --num-images 4 \
    --output-dir ./visual_T4111CE4 --start-idx 3306 --overlay-alpha 0.7