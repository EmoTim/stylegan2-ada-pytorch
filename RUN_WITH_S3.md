# Running generate.py with S3 Upload

## What Changed

I've modified [generate.py](generate.py) to automatically upload images to S3 as they're generated.

**New features:**
- Added `--s3-bucket` parameter
- Images are saved locally AND uploaded to S3 automatically
- Uses in-memory buffer for efficient uploads
- Progress messages show S3 upload status

## Your Command with S3 Upload

Based on your original command, here's how to run it with S3 upload:

```bash
uv run python generate.py \
    --outdir=out \
    --seeds=0-1000000 \
    --weight-vector=weight.npy \
    --style-range 0 4 \
    --alphas -5:5:10 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
    --vgg-path=/home/sagemaker-user/stylegan2-ada-pytorch/dex_age_classifier.pth \
    --s3-bucket=s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images
```

**Note:** I noticed your original command had `--4` which seems like a typo. I assumed you meant `--seeds=0-1000000` for 1M images.

## Test First with Small Batch

Before running 1M images, test with a small batch:

```bash
uv run python generate.py \
    --outdir=out \
    --seeds=0-10 \
    --weight-vector=weight.npy \
    --style-range 0 4 \
    --alphas -5:5:10 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
    --vgg-path=dex_age_classifier.pth \
    --s3-bucket=s3://emobot-prod-workspace-bucket/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images
```

This will generate only 11 images (seeds 0-10) and upload them to S3.

## What Happens

1. **Local Save**: Images are saved to `out/` directory locally
2. **S3 Upload**: Each image is immediately uploaded to S3
3. **Directory Structure**:
   - With weight vectors: `out/alpha_{value}/seed{XXXX}_styles0-4.png`
   - Without: `out/seed{XXXX}.png`
4. **S3 Structure**: Same structure as local, under your specified S3 path

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

## Running Without S3

If you want to skip S3 upload, just omit the `--s3-bucket` parameter:
```bash
uv run python generate.py \
    --outdir=out \
    --seeds=0-10 \
    --weight-vector=weight.npy \
    --style-range 0 4 \
    --alphas -5:5:10 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
    --vgg-path=dex_age_classifier.pth
```

## Important Notes

1. **AWS Credentials**: Make sure you're logged in with `aws sso login --profile Timothe`
2. **VGG Path**: Update `--vgg-path` to the correct local path if needed
3. **Age Filtering**: Your code filters out faces age ≤ 20 and faces with no detection
4. **Storage**: 1M images × ~1MB each ≈ 1TB of storage needed
5. **Time**: With age filtering, expect this to take many hours/days depending on your GPU

## Troubleshooting

### Error: "No such file 'dex_age_classifier.pth'"
Update the path to where your VGG model is located:
```bash
--vgg-path=/path/to/dex_age_classifier.pth
```

### Error: "Access Denied" to S3
Check your AWS credentials:
```bash
aws sts get-caller-identity --profile Timothe
```

### Error: "No module named 'boto3'"
Install it:
```bash
uv pip install boto3
```

## Performance Tips

For 1M images:
1. **Run in background** with `nohup` or `screen`
2. **Monitor progress** with `tail -f` on output
3. **Check S3** periodically to verify uploads
4. **Local cleanup**: Delete local files after confirming S3 upload

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
