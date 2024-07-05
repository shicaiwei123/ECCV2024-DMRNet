#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 0 --beta 0.1 0.547
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 0 --beta 0.5 0.579


#python main.py --train --ckpt_path results --alpha 0 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 0 --beta 0.0 0.527
#python main.py --train --ckpt_path results --alpha 0 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 0 --beta 0.0 --use_video_frames 3  555




#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.1 --use_video_frames 3 59.1
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.5 --use_video_frames 3 62.1
#
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.1 --use_video_frames 1  53.9
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.5 --use_video_frames 1 58.4
#
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.05 --use_video_frames 1 53.5


#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.3 --use_video_frames 1 0.563
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.6 --use_video_frames 1 0.579
#python main.py --train --ckpt_path results --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.8 --use_video_frames 1 0.574



#python main.py --train --ckpt_path results/ks/pme --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.3 --use_video_frames 1 --pme 1 56.8
#python main.py --train --ckpt_path results/ks/pme --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.5 --use_video_frames 1  --pme 1 56.7
#python main.py --train --ckpt_path results/ks/pme --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.8 --use_video_frames 1  --pme 1 56.6



export CUDA_VISIBLE_DEVICES=1
python main.py --train --ckpt_path results/ks/pme_share --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.3 --use_video_frames 1 --pme 1 57.1
python main.py --train --ckpt_path results/ks/pme_share --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.5 --use_video_frames 1  --pme 1  57.4
python main.py --train --ckpt_path results/ks/pme_share --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.8 --use_video_frames 1  --pme 1  55.8


python main_swin.py --train --ckpt_path results/ks_swin/full --dataset KineticSound --alpha 0 --dataset CREMAD --modulation Normal --pe 0 --gpu_ids 0 --beta 0 --pme 0   results/ks_swin/full/best_model_of_dataset_CREMAD_Normal_beta_0.0_pe_0_optimizer_sgd_modulate_starts_0_ends_50_epoch_91_acc_0.6519886363636364.pth.