#python main_shaspec.py --train --ckpt_path results/cramed/shaspec --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --dao_weight 0.1  --dco_weight 0.02 --unimodal_weight 0.0 53.8
#python main_shaspec.py --train --ckpt_path results/cramed/shaspec --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --dao_weight 0.5  --dco_weight 0.02 --unimodal_weight 0.0 54.2
#python main_shaspec.py --train --ckpt_path results/cramed/shaspec --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --dao_weight 1.0  --dco_weight 0.02 --unimodal_weight 0.0 52.7



export CUDA_VISIBLE_DEVICES=1
python main_shaspec.py --train --ckpt_path results/ks/shaspec --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --dao_weight 1.0  --dco_weight 0.02 --unimodal_weight 2.0
python main_shaspec.py --train --ckpt_path results/ks/shaspec --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --dao_weight 0.5  --dco_weight 0.02 --unimodal_weight 2.0
python main_shaspec.py --train --ckpt_path results/ks/shaspec --alpha 1e-3 --dataset KineticSound --modulation Normal --pe 0 --gpu_ids 1 --dao_weight 0.1  --dco_weight 0.02 --unimodal_weight 2.0