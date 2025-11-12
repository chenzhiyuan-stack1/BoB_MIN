python3 diffusion/train.py \
    --mode offline \
    --name offline_exp \
    --group my_offline_tests \
    --dataset_paths BoB_012.pickle BoB_45.pickle BoB_67.pickle \
    --buffer_size 4000000 \
    --use_wandb