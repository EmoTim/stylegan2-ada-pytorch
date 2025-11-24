import boto3

bucket = "emobot-prod-workspace-bucket"
prefix = "usecases/emobot-research/datasets/stylegan2-generated-images_v2/images/"

s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")

count = 0
pages = 0

for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    contents = page.get("Contents", [])
    count += len(contents)
    pages += 1
    if pages % 100 == 0:
        print(f"Processed {pages} pages -> {count} objects")

print("Total images:", count)
