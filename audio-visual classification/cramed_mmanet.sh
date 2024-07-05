
export CUDA_VISIBLE_DEVICES=0
python main_distillation.py --train --ckpt_path results/cramed/mmanet_pme --alpha 1e-3 --dataset CREMAD --modulation Normal --pe 0 --teacher_path /home/deep/shicaiwei/OGM-GE_CVPR2022_PME/results/cramed/full/best_model_of_dataset_CREMAD_Normal_alpha_0.8_pe_0_beta0_optimizer_sgd_modulate_starts_0_ends_50_epoch_29_acc_0.5951704545454546.pth --gpu 0