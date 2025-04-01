torchrun --nproc_per_node 2 -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/home/lee/PycharmProjects/stageCLIP/da-clip/src/training/datasets/universal/daclip_train.csv"  \
    --val-data="/home/lee/PycharmProjects/stageCLIP/da-clip/src/training/datasets/universal/daclip_val.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 100 \
    --batch-size=32 \
    --lr=3e-5 \
    --wd=0.05 \
    --model daclip_ViT-B-32 \
    --epochs=30 \
    --workers=8 \
    --da

#--model daclip_ViT-B-32 \
#   --name "stageClip_ViT-B-32-2023-09_b768x4_lr3e-5_e100_zeroaddd_multi_1" \
#    --pretrained "laion2b_s34b_b79k" \