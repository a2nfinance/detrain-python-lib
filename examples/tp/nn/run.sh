echo "Launching TP training"
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --rdzv_id=101 --rdzv_endpoint="localhost:9999" main.py --gpu="0_0_0" --epochs=4 --batch_size=50 --lr=0.001 --model_name="tp_01"
