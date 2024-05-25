echo "Launching FSDP + TP training"
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=101 --rdzv_endpoint="localhost:5972" main.py --epochs=4 --batch_size=50 --lr=0.001
