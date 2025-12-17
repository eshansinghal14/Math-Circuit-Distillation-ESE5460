import boto3
from constants import BUCKET_NAME

s3 = boto3.client("s3")
model_name = 'model_2000'
obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)