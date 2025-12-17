"""
Circuit Distillation Training

Implements post-training with composite loss:
    L_total = L_CE(y, ŷ_s) + λ * Σ L_CKA(paired_heads)

This trains the student model to both perform the task correctly (CE loss)
and align its internal circuit representations with the teacher (CKA loss).
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login

# Local imports
from cka_loss import CKALoss, MultiHeadCKALoss
from head_pairing import load_ablation_scores, create_head_mapping, get_paired_indices

try:
    from constants import HF_TOKEN, BUCKET_NAME
except ImportError:
    HF_TOKEN = os.environ.get('HF_TOKEN', '')
    BUCKET_NAME = os.environ.get('S3_BUCKET', 'circuit-distillation')


@dataclass
class TrainingConfig:
    """Configuration for circuit distillation training."""
    # Models
    teacher_model: str = 'meta-llama/Meta-Llama-3-8B'
    student_model: str = 'meta-llama/Llama-3.2-1B'
    
    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # CKA loss
    lambda_cka: float = 0.5  # Weight for CKA loss
    top_k_heads: Optional[int] = None  # If set, only align top-k heads
    
    # Data
    dataset_path: str = '../datasets/2d_add_train_80.json'
    max_seq_len: int = 16
    
    # Checkpointing
    save_dir: str = '../checkpoints'
    save_every: int = 100
    
    # Device
    device: str = 'auto'
    use_fp16: bool = True


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic problems."""
    
    def __init__(self, data_path: str, tokenizer, max_len: int = 16):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.prompts = list(self.data.keys())
        self.answers = [str(self.data[p]) for p in self.prompts]
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        answer = self.answers[idx]
        
        # Full sequence: prompt + answer
        full_text = prompt + answer
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels: mask out prompt tokens with -100
        prompt_encoding = self.tokenizer(prompt, return_tensors='pt')
        prompt_len = prompt_encoding['input_ids'].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ActivationCache:
    """Caches activations from specific model layers during forward pass."""
    
    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []
    
    def create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook(module, inputs, output):
            # Handle different output formats
            if isinstance(output, tuple):
                output = output[0]
            self.activations[layer_idx] = output.detach()
        return hook
    
    def register_hooks(self, model, layer_indices: List[int], component: str = 'mlp'):
        """Register hooks on specified layers.
        
        Args:
            model: The transformer model
            layer_indices: Which layer indices to hook
            component: 'mlp', 'attention', or 'output'
        """
        self.clear()
        
        for idx in layer_indices:
            if component == 'mlp':
                # Hook the MLP output
                layer = model.model.layers[idx].mlp
                hook = layer.register_forward_hook(self.create_hook(idx))
            elif component == 'attention':
                # Hook the attention output
                layer = model.model.layers[idx].self_attn
                hook = layer.register_forward_hook(self.create_hook(idx))
            elif component == 'output':
                # Hook the full layer output
                layer = model.model.layers[idx]
                hook = layer.register_forward_hook(self.create_hook(idx))
            else:
                raise ValueError(f"Unknown component: {component}")
            
            self.hooks.append(hook)
    
    def clear(self):
        """Clear cached activations and remove hooks."""
        self.activations = {}
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get cached activations."""
        return self.activations


class CircuitDistillationTrainer:
    """Main trainer for circuit distillation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Login to HuggingFace
        if HF_TOKEN:
            login(HF_TOKEN)
        
        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        # Load models
        print(f"Loading teacher model: {config.teacher_model}")
        self.teacher = self._load_model(config.teacher_model, trainable=False)
        
        print(f"Loading student model: {config.student_model}")
        self.student = self._load_model(config.student_model, trainable=True)
        
        # Setup head mapping
        self._setup_head_mapping()
        
        # Setup activation caches
        self.teacher_cache = ActivationCache()
        self.student_cache = ActivationCache()
        
        # Setup losses
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.cka_loss = MultiHeadCKALoss(reduction='mean')
        
        # Setup optimizer (only for student)
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, config.dataset_path)
        self.dataset = ArithmeticDataset(dataset_path, self.tokenizer, config.max_seq_len)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Scheduler
        total_steps = len(self.dataloader) * config.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # Metrics
        self.metrics_history = []
    
    def _load_model(self, model_name: str, trainable: bool = True):
        """Load a model with appropriate settings."""
        dtype = torch.float16 if self.config.use_fp16 else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map='auto' if self.device.type == 'cuda' else None,
            low_cpu_mem_usage=True
        )
        
        if not trainable:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        
        return model
    
    def _setup_head_mapping(self):
        """Setup the mapping between student and teacher heads."""
        # Try to load from ablation cache
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(script_dir, '..', 'results', 'ablation_cache.json')
        
        if os.path.exists(cache_path):
            print(f"Loading head mapping from {cache_path}")
            self.head_mapping = get_paired_indices(
                cache_path, 
                top_k=self.config.top_k_heads
            )
            print(f"Created {len(self.head_mapping)} head pairs")
        else:
            # Default: map layers proportionally
            print("No ablation cache found, using proportional layer mapping")
            student_layers = self.student.config.num_hidden_layers
            teacher_layers = self.teacher.config.num_hidden_layers
            
            self.head_mapping = {}
            for s_idx in range(student_layers):
                t_idx = int(s_idx * teacher_layers / student_layers)
                self.head_mapping[s_idx] = t_idx
        
        # Get layer indices for hooking
        self.student_layers = list(self.head_mapping.keys())
        self.teacher_layers = list(set(self.head_mapping.values()))
    
    def _compute_loss(
        self, 
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        student_acts: Dict[int, torch.Tensor],
        teacher_acts: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute composite loss: CE + λ * CKA"""
        
        # Cross-entropy loss
        # Shift logits and labels for next-token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # CKA loss over paired heads
        cka_total, cka_scores = self.cka_loss(
            student_acts,
            teacher_acts,
            self.head_mapping
        )
        
        # Composite loss
        total_loss = ce + self.config.lambda_cka * cka_total
        
        metrics = {
            'ce_loss': ce.item(),
            'cka_loss': cka_total.item(),
            'total_loss': total_loss.item(),
            'mean_cka': sum(cka_scores.values()) / len(cka_scores) if cka_scores else 0.0
        }
        
        return total_loss, metrics
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Register hooks
        self.student_cache.register_hooks(self.student, self.student_layers, 'mlp')
        self.teacher_cache.register_hooks(self.teacher, self.teacher_layers, 'mlp')
        
        try:
            # Teacher forward (no grad)
            with torch.no_grad():
                self.teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_acts = self.teacher_cache.get_activations()
            
            # Student forward
            student_outputs = self.student(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            student_acts = self.student_cache.get_activations()
            
            # Compute loss
            loss, metrics = self._compute_loss(
                student_outputs.logits,
                labels,
                student_acts,
                teacher_acts
            )
            
            # Backward
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
        finally:
            # Clean up hooks
            self.student_cache.clear()
            self.teacher_cache.clear()
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.student.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            metrics = self.train_step(batch)
            
            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.dataloader)} - "
                      f"CE: {metrics['ce_loss']:.4f}, "
                      f"CKA: {metrics['cka_loss']:.4f}, "
                      f"Mean CKA: {metrics['mean_cka']:.4f}")
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return dict(epoch_metrics)
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Lambda CKA: {self.config.lambda_cka}")
        print(f"  Head pairs: {len(self.head_mapping)}")
        print()
        
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            metrics = self.train_epoch(epoch)
            metrics['epoch'] = epoch + 1
            self.metrics_history.append(metrics)
            
            print(f"  Epoch {epoch + 1} complete - "
                  f"CE: {metrics['ce_loss']:.4f}, "
                  f"CKA: {metrics['cka_loss']:.4f}, "
                  f"Mean CKA: {metrics['mean_cka']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final save
        self.save_checkpoint('final')
        self.save_metrics()
        
        print("\nTraining complete!")
    
    def save_checkpoint(self, tag):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f'student_epoch_{tag}')
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  Saved checkpoint to {path}")
    
    def save_metrics(self):
        """Save training metrics."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"  Saved metrics to {path}")
    
    @torch.no_grad()
    def evaluate(self, data_path: str = None) -> Dict:
        """Evaluate the student model."""
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, '..', 'datasets', '2d_add_test_20.json')
        
        with open(data_path, 'r') as f:
            test_data = json.load(f)
        
        self.student.eval()
        
        correct = 0
        total = 0
        
        prompts = list(test_data.keys())
        answers = [test_data[p] for p in prompts]
        
        for prompt, answer in zip(prompts, answers):
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            outputs = self.student.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract predicted answer
            try:
                pred = int(response.split('=')[-1].strip())
                if pred == answer:
                    correct += 1
            except:
                pass
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }


def main():
    parser = argparse.ArgumentParser(description='Circuit Distillation Training')
    
    # Model args
    parser.add_argument('--teacher', type=str, default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--student', type=str, default='meta-llama/Llama-3.2-1B')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda_cka', type=float, default=0.5)
    
    # Data args
    parser.add_argument('--dataset', type=str, default='../datasets/2d_add_train_80.json')
    
    # Output args
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        teacher_model=args.teacher,
        student_model=args.student,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_cka=args.lambda_cka,
        dataset_path=args.dataset,
        save_dir=args.save_dir
    )
    
    trainer = CircuitDistillationTrainer(config)
    
    # Initial evaluation
    print("Initial evaluation...")
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")
    
    # Train
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")


if __name__ == '__main__':
    main()
