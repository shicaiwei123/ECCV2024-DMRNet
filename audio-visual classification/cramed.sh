
#python main.py --train --ckpt_path results/cramed/pme --alpha 0 --dataset CREMAD --modulation Normal --pe 0 --gpu_ids 1 --beta 0 --pme 0



# python main.py --train --ckpt_path results/cramed/pme --alpha 1e-3 --dataset CREMAD --modulation Normal --pe 0 --gpu_ids 1 --beta 0.1 --pme 1 60.61
# python main.py --train --ckpt_path results/cramed/pme --alpha 1e-3 --dataset CREMAD --modulation Normal --pe 0 --gpu_ids 1 --beta 0.15 --pme 1 59.9
python main.py --train --ckpt_path results/cramed/pme --alpha 1e-3 --dataset CREMAD --modulation Normal --pe 0 --gpu_ids 1 --beta 0.20 --pme 1 64.2

python main.py --train --ckpt_path results/cramed/pme --alpha 1e-3 --dataset CREMAD --modulation Normal --pe 0 --gpu_ids 1 --beta 0.25 --pme 1 63.8

