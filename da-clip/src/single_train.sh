python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/home/lee/PycharmProjects/stageCLIP/da-clip/src/training/datasets/universal/daclip_train.csv"  \
    --val-data="/home/lee/PycharmProjects/stageCLIP/da-clip/src/training/datasets/universal/daclip_val.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 100 \
    --batch-size=16 \
    --name "stageCLIP_ViT-B-32-2023-09_b512x1_lr2e-5_e30_test_10" \
    --lr=2e-5 \
    --wd=0.05 \
    --model daclip_ViT-B-32 \
    --epochs=30 \
    --workers=8 \
    --da

#    --model daclip_ViT-B-32 \

#      --name "daclip_ViT-B-32-2023-09_b512x1_lr2e-5_e30_test_10" \
#    --pretrained "laion2b_s34b_b79k" \
