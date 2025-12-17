#!/bin/bash
# EC2 Setup Script for Circuit Distillation Training
# 
# This script sets up an EC2 instance for training.
# Run this after launching an instance with the Deep Learning AMI.
#
# Recommended instances:
#   - p4d.24xlarge: 8x A100 40GB (~$32/hr)
#   - p5.48xlarge: 8x H100 80GB (~$98/hr)
#   - g5.12xlarge: 4x A10G 24GB (~$5.70/hr) for testing

set -e

echo "=========================================="
echo "Circuit Distillation EC2 Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -qq

# Activate PyTorch environment (Deep Learning AMI)
if [ -d "/opt/conda" ]; then
    echo "Activating conda environment..."
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    boto3>=1.34.0 \
    sagemaker>=2.200.0 \
    datasets>=2.15.0 \
    wandb>=0.16.0 \
    tqdm>=4.66.0

# Configure HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN environment variable not set."
    echo "You'll need to set this to download Llama models:"
    echo "  export HF_TOKEN=your_token_here"
    echo ""
fi

# Configure AWS credentials if not already done
if ! aws sts get-caller-identity &>/dev/null; then
    echo ""
    echo "AWS credentials not configured. Run:"
    echo "  aws configure"
    echo ""
fi

# Clone repository if not present
REPO_DIR="Math-Circuit-Distillation-ESE5460"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    # Replace with your repo URL
    git clone https://github.com/YOUR_USERNAME/Math-Circuit-Distillation-ESE5460.git
fi

cd $REPO_DIR

# Create necessary directories
mkdir -p checkpoints results

# Download datasets from S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    echo "Downloading datasets from S3..."
    aws s3 sync s3://$S3_BUCKET/circuit-distillation/datasets ./datasets
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  cd $REPO_DIR"
echo "  python src/circuit_distillation.py \\"
echo "      --teacher meta-llama/Meta-Llama-3-8B \\"
echo "      --student meta-llama/Llama-3.2-1B \\"
echo "      --lambda_cka 0.5 \\"
echo "      --epochs 100"
echo ""
echo "For multi-GPU training with accelerate:"
echo "  accelerate launch --num_processes 8 src/circuit_distillation.py ..."
echo ""
