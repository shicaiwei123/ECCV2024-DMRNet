
export CUDA_VISIBLE_DEVICES=1
python main_distillation.py --train --ckpt_path results/ks/mmanet_pme --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --teacher_path /home/deep/shicaiwei/OGM-GE_CVPR2022_PME/results/ks/full/best_model_of_dataset_KineticSound_Normal_alpha_0.8_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_99_acc_0.64125.pth --gpu 1 --pme 1