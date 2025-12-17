# AWS Setup for Circuit Distillation Training

This directory contains scripts and configurations for running circuit distillation training on AWS with A100/H100 GPUs.

## Quick Start

### Option 1: SageMaker (Recommended)

SageMaker provides managed infrastructure with easy scaling.

```bash
# 1. Configure AWS credentials
aws configure

# 2. Create S3 bucket for data and results
aws s3 mb s3://your-bucket-name

# 3. Upload training data
aws s3 sync ../datasets s3://your-bucket-name/circuit-distillation/datasets

# 4. Run training
python train_sagemaker.py \
    --bucket your-bucket-name \
    --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
```

### Option 2: EC2 Direct

For more control and potentially lower costs.

```bash
# 1. Launch instance (see ec2_setup.sh for details)
./ec2_setup.sh

# 2. SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Run training
python src/circuit_distillation.py \
    --teacher meta-llama/Meta-Llama-3-8B \
    --student meta-llama/Llama-3.2-1B \
    --lambda_cka 0.5 \
    --epochs 100
```

## Instance Types

| Instance | GPUs | GPU Memory | Cost/hr | Recommended For |
|----------|------|------------|---------|-----------------|
| `ml.g5.xlarge` | 1x A10G | 24GB | ~$1.20 | Testing, small experiments |
| `ml.g5.12xlarge` | 4x A10G | 96GB | ~$5.70 | Medium experiments |
| `ml.p4d.24xlarge` | 8x A100 | 320GB | ~$32.00 | Full training |
| `ml.p5.48xlarge` | 8x H100 | 640GB | ~$98.00 | Large-scale training |

## Files

- `train_sagemaker.py` - SageMaker training script launcher
- `ec2_setup.sh` - EC2 instance setup script
- `requirements_aws.txt` - AWS-specific dependencies
