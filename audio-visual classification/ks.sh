

export CUDA_VISIBLE_DEVICES=1
python main.py --train --ckpt_path results/ks/pme_share --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --beta 0.3 --use_video_frames 1 --pme 1 56.8



