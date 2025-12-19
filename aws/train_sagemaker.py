import os
import argparse
from datetime import datetime

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput


def get_execution_role(role_arn: str = None):
    if role_arn:
        return role_arn
    
    try:
        return sagemaker.get_execution_role()
    except:
        raise ValueError(
            "Could not determine SageMaker execution role. "
            "Please provide --role argument with your IAM role ARN."
        )


def launch_training(
    bucket: str,
    role: str,
    instance_type: str = 'ml.p4d.24xlarge',
    instance_count: int = 1,
    epochs: int = 100,
    batch_size: int = 32,
    lambda_cka: float = 0.5,
    learning_rate: float = 1e-5,
    teacher_model: str = 'meta-llama/Meta-Llama-3-8B',
    student_model: str = 'meta-llama/Llama-3.2-1B',
    job_name: str = None,
    spot_instances: bool = False,
    max_wait: int = 86400,  
):
    session = sagemaker.Session()
    region = session.boto_region_name
    
    print(f"Region: {region}")
    print(f"Bucket: {bucket}")
    print(f"Instance: {instance_type} x {instance_count}")
    
    if job_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        job_name = f"circuit-distillation-{timestamp}"
    
    s3_prefix = f"s3://{bucket}/circuit-distillation"
    output_path = f"{s3_prefix}/output"
    checkpoint_path = f"{s3_prefix}/checkpoints/{job_name}"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.dirname(script_dir)
    
    hyperparameters = {
        'teacher': teacher_model,
        'student': student_model,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': learning_rate,
        'lambda_cka': lambda_cka,
        'save_dir': '/opt/ml/checkpoints',
        'dataset': '/opt/ml/input/data/train/2d_add_train_80.json'
    }
    
    environment = {
        'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
        'TRANSFORMERS_CACHE': '/opt/ml/input/data/cache'
    }
    
    estimator = PyTorch(
        entry_point='circuit_distillation.py',
        source_dir=os.path.join(source_dir, 'src'),
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_path,
        checkpoint_local_path='/opt/ml/checkpoints',
        use_spot_instances=spot_instances,
        max_wait=max_wait if spot_instances else None,
        max_run=max_wait,
        disable_profiler=True,
        debugger_hook_config=False,
    )
    
    train_input = TrainingInput(
        s3_data=f"{s3_prefix}/datasets",
        s3_data_type='S3Prefix',
        input_mode='File'
    )
    
    print(f"\nLaunching training job: {job_name}")
    print(f"Output: {output_path}")
    
    estimator.fit(
        inputs={'train': train_input},
        job_name=job_name,
        wait=False  
    )
    
    print(f"\nTraining job submitted!")
    print(f"Monitor at: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    
    return job_name


def main():
    parser = argparse.ArgumentParser(description='Launch SageMaker training job')
    
    parser.add_argument('--bucket', type=str, required=True,
                        help='S3 bucket name')
    parser.add_argument('--role', type=str, default=None,
                        help='SageMaker execution role ARN')
    
    parser.add_argument('--instance_type', type=str, default='ml.p4d.24xlarge',
                        choices=['ml.g5.xlarge', 'ml.g5.12xlarge', 'ml.p4d.24xlarge', 'ml.p5.48xlarge'],
                        help='Instance type')
    parser.add_argument('--instance_count', type=int, default=1,
                        help='Number of instances')
    parser.add_argument('--spot', action='store_true',
                        help='Use spot instances')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda_cka', type=float, default=0.5)
    
    parser.add_argument('--teacher', type=str, default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--student', type=str, default='meta-llama/Llama-3.2-1B')
    
    parser.add_argument('--job_name', type=str, default=None)
    
    args = parser.parse_args()
    
    role = get_execution_role(args.role)
    
    launch_training(
        bucket=args.bucket,
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_cka=args.lambda_cka,
        learning_rate=args.lr,
        teacher_model=args.teacher,
        student_model=args.student,
        job_name=args.job_name,
        spot_instances=args.spot,
    )


if __name__ == '__main__':
    main()
