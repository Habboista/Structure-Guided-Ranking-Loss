CUDA_VISIBLE_DEVICES=0 python3 demo.py \
    --img-size 448 448 \
    --data-dir input \
    --result-dir output \
    --checkpoint model.pth.tar \
    --gpu-id 0
