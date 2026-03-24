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


  
# CUDA_VISIBLE_DEVICES=0 python visualize_expert_assignment.py \
#   -c conf/cifar100/2_512_300E_t41_Const_E2_v2.yml \
#     --resume ./output/260320-2_512_300E_t41_Const_E2_v2/model_best.pth.tar \
#     --num-images 4 \
#     --output-dir ./visual/T41CE2 --start-idx 3306 --overlay-alpha 0.7

#   CUDA_VISIBLE_DEVICES=0 python visualize_expert_assignment.py \
#   -c conf/cifar100/2_512_300E_t4111_Const_E4_v2.yml \
#     --resume ./output/260320-2_512_300E_t4111_Const_E4_v2/model_best.pth.tar \
#     --num-images 4 \
#     --output-dir ./visual/T4111CE4 --start-idx 3306 --overlay-alpha 0.7

#   CUDA_VISIBLE_DEVICES=0 python visualize_expert_assignment.py \
#   -c conf/cifar100/2_512_300E_t4111_Const_E8_v2.yml \
#     --resume ./output/260320-2_512_300E_t4111_Const_E8_v2/model_best.pth.tar \
#     --num-images 4 \
#     --output-dir ./visual/T4111CE8 --start-idx 3306 --overlay-alpha 0.7


# CUDA_VISIBLE_DEVICES=1 python compute_tic.py -c output/260320-2_512_300E_t41_Const_E2_v2/args.yaml \
#   --resume output/260320-2_512_300E_t41_Const_E2_v2/model_best.pth.tar --output-dir ./tic_vis/e2 --num-images 3000

#   python compute_tic.py -c output/260320-2_512_300E_t4111_Const_E4_v2/args.yaml \
#   --resume output/260320-2_512_300E_t4111_Const_E4_v2/model_best.pth.tar --output-dir ./tic_vis --num-images 3000

#   CUDA_VISIBLE_DEVICES=2 python compute_tic.py -c output/260320-2_512_300E_t4111_Const_E8_v2/args.yaml \
#   --resume output/260320-2_512_300E_t4111_Const_E8_v2/model_best.pth.tar --output-dir ./tic_vis/e8 --num-images 3000


############################
#260323 시각화 결과가 계속 이상해서 분석. --> 결론은 E4부터 expert가 고르는 토큰 중요도가 확실히 떨어진다고 판단. mlp ratio가 너무 줄어서 그런건가?라는 생각이 듦.
# 추가 실험은 mlp ratio를 2.0으로 증가시켜서 실험해보자.
#아래는 log.
#2026-03-21 01:18:57,412 INFO: *** Best metric: 79.45 (epoch 286)
  python eval_routing_masks.py \
    -c  output/260320-2_512_300E_t41_Const_E2_v2/args.yaml\
    --resume output/260320-2_512_300E_t41_Const_E2_v2/model_best.pth.tar

=== Routing-mask accuracy comparison (CIFAR-100 eval set) ===
Case A   (T=4 expert only)          : 0.7881
Case A'  (T=4 expert, half tokens)  : 0.7730
Case B   (non T=4 experts only)     : 0.4426
Case C   (random, |C| = |A|)        : 0.7827
Case C'  (random, |C'| = |A'|)      : 0.7406

=== Average token assignment ratios (last block) ===
Mean ratio T=4 expert tokens      : 0.6039
Mean ratio non T=4 expert tokens  : 0.3961
Mean overlap (A ∩ C) / |A|          : 0.6041  (A = T=4 expert tokens, C = random tokens)



#2026-03-21 01:47:44,016 INFO: *** Best metric: 78.71 (epoch 303)
  python eval_routing_masks.py \
    -c  output/260320-2_512_300E_t4111_Const_E4_v2/args.yaml\
    --resume output/260320-2_512_300E_t4111_Const_E4_v2/model_best.pth.tar
=== Routing-mask accuracy comparison (CIFAR-100 eval set) ===
Case A   (T=4 expert only)          : 0.7651
Case A'  (T=4 expert, half tokens)  : 0.7019
Case B   (non T=4 experts only)     : 0.7581
Case C   (random, |C| = |A|)        : 0.7719
Case C'  (random, |C'| = |A'|)      : 0.6089

=== Average token assignment ratios (last block) ===
Mean ratio T=4 expert tokens      : 0.4245
Mean ratio non T=4 expert tokens  : 0.5755



#2026-03-21 02:12:08,096 INFO: *** Best metric: 78.78 (epoch 302)
  python eval_routing_masks.py \
    -c  output/260320-2_512_300E_t4111_Const_E8_v2/args.yaml\
    --resume output/260320-2_512_300E_t4111_Const_E8_v2/model_best.pth.tar


    === Routing-mask accuracy comparison (CIFAR-100 eval set) ===
Case A   (T=4 expert only)          : 0.7625
Case A'  (T=4 expert, half tokens)  : 0.6257
Case B   (non T=4 experts only)     : 0.7483
Case C   (random, |C| = |A|)        : 0.6931
Case C'  (random, |C'| = |A'|)      : 0.1402

=== Average token assignment ratios (last block) ===
Mean ratio T=4 expert tokens      : 0.2728
Mean ratio non T=4 expert tokens  : 0.7272
Mean overlap (A ∩ C) / |A|          : 0.2719  (A = T=4 expert tokens, C = random tokens)






# 바로 위에 대한 추가 실험. mlp ratio를 2.0으로 증가시켜서 실험함.
#79.18% 로 종료
CUDA_VISIBLE_DEVICES=1 python train.py -c conf/cifar100/2_512_300E_t4111_R1_E4.yml -data-dir /dataset/CIFAR100/ --model sdt --spike-mode lif


# 시각화 이쁘게 잘나옴
  CUDA_VISIBLE_DEVICES=1 python visualize_expert_assignment.py \
  -c conf/cifar100/2_512_300E_t4111_R2_E4.yml \
    --resume ./output/260323-2_512_300E_t4111_R2_E4/model_best.pth.tar \
    --num-images 4 \
    --output-dir ./visual/T4111R2E4 --start-idx 3306 --overlay-alpha 0.7

# 이 결과도 괜찮음.
  CUDA_VISIBLE_DEVICES=1 python eval_routing_masks.py \
    -c  output/260323-2_512_300E_t4111_R2_E4/args.yaml\
    --resume output/260323-2_512_300E_t4111_R2_E4/model_best.pth.tar


=== Routing-mask accuracy comparison (CIFAR-100 eval set) ===
Case A   (T=4 expert only)          : 0.7860
Case A'  (T=4 expert, half tokens)  : 0.7605
Case B   (non T=4 experts only)     : 0.6604
Case C   (random, |C| = |A|)        : 0.7644
Case C'  (random, |C'| = |A'|)      : 0.4590

=== Average token assignment ratios (last block) ===
Mean ratio T=4 expert tokens      : 0.3841
Mean ratio non T=4 expert tokens  : 0.6159
Mean overlap (A ∩ C) / |A|          : 0.3824  (A = T=4 expert tokens, C = random tokens)

#결론 -> mlp_ratio를 2 아래로 내리면 expert가 너무 약해짐.




#########################
python run_temporal_merging_experiments.py \
  -c conf/gesture/2_256_200E_t101_Const_E2.yml \
  --resume output/260324-2_256_200E_t101_Const_E2_NewRouter/model_best.pth.tar \
  --device cuda:3 \
  --experiment-name gesture_temporal_merge \
  --max-batches 16 \
  --compute-isi