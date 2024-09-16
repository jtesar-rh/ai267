import io
import os
import boto3
key_id = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")
endpoint = os.getenv("AWS_S3_ENDPOINT")
bucket = 'datasets'
s3 = boto3.client(
    "s3",
    region,
    aws_access_key_id=key_id,
    aws_secret_access_key=secret_key,
    endpoint_url=endpoint,
    use_ssl=False
)

# retrieve the metadata of contents within the bucket
objects = s3.list_objects_v2(Bucket=bucket)

# output the name of each object within the bucket
for obj in objects["Contents"]:
    print(obj["Key"])
    s3.download_file(bucket, obj["Key"], obj["Key"], )