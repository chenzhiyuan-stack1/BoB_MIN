python3 diffusion/train.py \
    --mode online \
    --name online_exp_1 \
    --group my_online_tests \
    --online_rounds 10 \
    --iterations_per_round 10000 \
    --batch_size 2048 \
    --lr 1e-5 \
    --use_wandb