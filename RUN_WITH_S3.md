# Running generate.py with S3 Upload


```bash
uv run python generate.py \
    --outdir=out \
    --seeds=0-20000 \
    --weight-vector=weight.npy \
    --style-range 0 4 \
    --alphas -5:5:51 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
    --vgg-path=/home/sagemaker-user/stylegan2-ada-pytorch/dex_age_classifier.pth \
    --s3-bucket=s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images
```

Example with nohup:
```bash
nohup uv run python generate.py \
    --outdir=out \
    --seeds=0-1000000 \
    --weight-vector=weight.npy \
    --style-range 0 4 \
    --alphas -5:5:10 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
    --vgg-path=dex_age_classifier.pth \
    --s3-bucket=s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images \
    > generation.log 2>&1 &
```

Then monitor with:
```bash
tail -f generation.log
```

## Expected Output

You'll see output like:
```
Loading networks from "https://..."
S3 upload enabled: s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images/
Initializing age predictor...
Generating image for seed 0 (0/11) ...
  ✓ Seed 0: Age 25.3 > 20, proceeding...
  ✓ Uploaded to s3://emobot-prod-workspace-bucket/.../images/alpha_-5.0/seed0000_styles0-4.png
  ✓ Uploaded to s3://emobot-prod-workspace-bucket/.../images/alpha_-3.9/seed0000_styles0-4.png
  ...
```

## S3 Location

Your images will be at:
```
s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images/
├── alpha_-5.0/
│   ├── seed0000_styles0-4.png
│   ├── seed0001_styles0-4.png
│   └── ...
├── alpha_-3.9/
│   ├── seed0000_styles0-4.png
│   └── ...
└── ...
```

## Download from S3

To download all generated images:
```bash
aws s3 sync s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images/ ./downloaded_images/ --profile Timothe
```



