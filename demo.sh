conda activate kornia
#messytable wh train foot
CUDA_VISIBLE_DEVICES=0,1 python -m pdb  main.py -d messytable --train_ratio .5026 --var_extrinsic_matrices --epochs 80 --wh_train foot

#messytable wh train center
CUDA_VISIBLE_DEVICES=0,1 python -m pdb  main.py -d messytable --train_ratio .5026 --var_extrinsic_matrices --epochs 80 --wh_train center

#sythretail wh train center
CUDA_VISIBLE_DEVICES=0,1 python -m pdb  main.py -d synthretail --train_ratio .5026 --var_extrinsic_matrices --epochs 80 --wh_train center
