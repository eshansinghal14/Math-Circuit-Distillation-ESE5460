import io
import torch
import boto3

from constants import BUCKET_NAME
from circuit_discovery.models import CircuitDiscoveryModel
from utils import load_custom_model

s3 = boto3.client("s3")

model_name = "model_2000"
model = load_custom_model(model_name)
model.eval()

def corr_distance_sanity_check():
    pass

