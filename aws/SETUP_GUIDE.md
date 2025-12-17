# AWS Training Setup Guide - Step by Step

## Prerequisites Checklist
- [ ] AWS CLI installed and configured
- [ ] HuggingFace account with Llama access
- [ ] SageMaker execution role created

---

## Step 1: Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it something like "circuit-distillation"
4. Select **"Read"** access
5. Click **"Generate"**
6. Copy the token (starts with `hf_...`)

> **Important**: You also need access to Llama models. Go to https://huggingface.co/meta-llama/Llama-3.2-1B and accept the license if you haven't.

---

## Step 2: Create SageMaker Execution Role

1. Go to **AWS Console** → Search for **"IAM"** → Click **IAM**
2. In the left sidebar, click **"Roles"**
3. Click **"Create role"** (blue button)
4. For **Trusted entity type**: Select **"AWS service"**
5. For **Use case**: 
   - Select **"SageMaker"** from the dropdown
   - Select **"SageMaker - Execution"**
   - Click **Next**
6. On **Add permissions** page:
   - Search and check: `AmazonSageMakerFullAccess`
   - Search and check: `AmazonS3FullAccess`
   - Click **Next**
7. For **Role name**: Enter `SageMakerCircuitDistillation`
8. Click **"Create role"**
9. Click on the role you just created
10. **Copy the ARN** (looks like `arn:aws:iam::123456789012:role/SageMakerCircuitDistillation`)

---

## Step 3: Install SageMaker SDK

Run in your terminal:

```bash
cd "/Users/vedantgaur/Downloads/Classes/Eng/ESE 5460/Math-Circuit-Distillation-ESE5460"
source venv/bin/activate
pip install sagemaker
```

---

## Step 4: Set Environment Variables

```bash
export HF_TOKEN="hf_your_token_here"
```

---

## Step 5: Launch Training

For a **test run** (cheaper, ~$1.20/hr):
```bash
python aws/train_sagemaker.py \
    --bucket math-circuit-distillation-ese5460 \
    --role "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerCircuitDistillation" \
    --instance_type ml.g5.xlarge \
    --epochs 5 \
    --lambda_cka 0.5
```

For **full training** with A100 (~$32/hr):
```bash
python aws/train_sagemaker.py \
    --bucket math-circuit-distillation-ese5460 \
    --role "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerCircuitDistillation" \
    --instance_type ml.p4d.24xlarge \
    --epochs 100 \
    --lambda_cka 0.5
```

---

## Step 6: Monitor Training

The script will output a URL to the SageMaker console. You can also:

```bash
# Check job status
aws sagemaker describe-training-job --training-job-name YOUR_JOB_NAME

# View CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

---

## Step 7: Download Results

After training completes:
```bash
aws s3 sync s3://math-circuit-distillation-ese5460/circuit-distillation/output ./output
```

---

## Cost Estimates

| Instance | GPU | Hourly Cost | 10 epochs (~1hr) | 100 epochs (~10hr) |
|----------|-----|-------------|------------------|-------------------|
| ml.g5.xlarge | 1x A10G | $1.20 | ~$1.20 | ~$12 |
| ml.p4d.24xlarge | 8x A100 | $32.77 | ~$33 | ~$330 |
| ml.p5.48xlarge | 8x H100 | $98.32 | ~$98 | ~$980 |

**Recommendation**: Start with `ml.g5.xlarge` for 5 epochs to verify everything works, then scale up.
