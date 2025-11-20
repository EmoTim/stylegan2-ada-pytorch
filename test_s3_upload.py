#!/usr/bin/env python3
"""
Quick test script to verify S3 upload works before running full generation
"""
import boto3
from PIL import Image
import numpy as np
import io

def test_s3_upload():
    """Test that we can upload to S3 bucket."""

    bucket = 'emobot-prod-workspace-bucket'
    prefix = 'usecases/emobot-research/datasets/stylegan2-generated-images_v2/images/test/'

    print("🧪 Testing S3 Upload...")
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}")

    try:
        # Create S3 client
        s3_client = boto3.client('s3')
        print("✓ S3 client created")

        # Create a test image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_img = Image.fromarray(img_array, 'RGB')
        print("✓ Test image created (256x256)")

        # Upload to S3
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        test_key = prefix + 'test_upload.png'
        s3_client.upload_fileobj(img_buffer, bucket, test_key)
        print(f"✓ Test image uploaded to s3://{bucket}/{test_key}")

        # Verify it exists
        response = s3_client.head_object(Bucket=bucket, Key=test_key)
        size_kb = response['ContentLength'] / 1024
        print(f"✓ Upload verified! Size: {size_kb:.2f} KB")

        # Clean up test file
        s3_client.delete_object(Bucket=bucket, Key=test_key)
        # print("✓ Test file cleaned up")
        print("✓ Test file kept for verification")

        print("\n✅ SUCCESS! S3 upload is working correctly.")
        print(f"\nYou can now run generate.py with:")
        print(f"--s3-bucket=s3://{bucket}/usecases/emobot-research/datasets/stylegan2-generated-images_v2/images")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPossible solutions:")
        print("1. Run: aws sso login --profile Timothe")
        print("2. Check bucket name and permissions")
        print("3. Verify AWS_PROFILE is set")
        return False


if __name__ == "__main__":
    test_s3_upload()
