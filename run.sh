

# torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint="localhost:0" sample.py --job.config_file=/home/hossamamer/ttt-video-dit/configs/eval/ttt-mlp/9s.toml --checkpoint.init_state_dir=/model_weights/vid_save_path2/ --eval.input_file=/home/hossamamer/ttt-video-dit/inputs/example-9s.json --eval.t5_model_dir=/home/hossamamer/.cache/huggingface/hub/models--google--t5-v1_1-xxl/snapshots/3db67ab1af984cf10548a73467f0e5bca2aaaeb2/ --wandb.disable


# --eval.vae_checkpoint_path=/workspace/CogVideoX-5B/vae/3d-vae.pt --wandb.disable


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
python sample.py --job.config_file ./configs/eval/ttt-mlp/9s.toml --eval.input_file=/home/hossamamer/ttt-video-dit/inputs/example-9s.json --checkpoint.init_state_dir=/model_weights/vid_save_path2/ --parallelism.dp_replicate=1 --parallelism.dp_sharding=1 --parallelism.tp_sharding=1 --wandb.disable

