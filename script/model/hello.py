import torch
import torch.nn as nn
from typing import Any, Dict, Union, Sequence
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from dataloader import get_train_dataset, get_val_dataset  # Import dataset functions
from swin_transformer import SwinTransformer  # Ensure this class only defines the model architecture
from custom_scheduler import WarmupCosineAnnealingScheduler  # Your custom scheduler

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class MyTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        # Initialize the model
        model_size = self.context.get_hparam("model_size")
        self.model = self.context.wrap_model(self.build_model(model_size))

        # Configure the optimizer
        optimizer = self.configure_optimizer(
            adamw_weight_decay_body=self.context.get_hparam("adamw_weight_decay_body"),
            adamw_weight_decay_head=self.context.get_hparam("adamw_weight_decay_head"),
            body_lr=self.context.get_hparam("body_lr"),
            head_lr=self.context.get_hparam("head_lr")
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)

        # Define the loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Calculate total training steps
        steps_per_epoch = self.context.get_hparam("steps_per_epoch")
        planned_epochs = self.context.get_hparam("full_run_epochs")
        self.total_steps = planned_epochs * steps_per_epoch
        self.warmup_steps = self.context.get_hparam("warmup_steps")

        # Define min_lrs per parameter group (order must match optimizer.param_groups)
        body_min_lr = self.context.get_hparam("body_min_lr")
        head_min_lr = self.context.get_hparam("head_min_lr")
        # Assuming four parameter groups as per configure_optimizer
        self.min_lrs = [body_min_lr, body_min_lr, head_min_lr, head_min_lr]

        # Initialize the custom scheduler without wrapping
        self.scheduler = WarmupCosineAnnealingScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lrs=self.min_lrs
        )

        # Initialize gradient accumulation counter
        self.accumulation_counter = 0
        self.thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

    def build_model(self, model_size):
        if model_size == "large":
            model = SwinTransformer(
                drop_rate=self.context.get_hparam("drop_rate"), 
                depths=[2,2,18,2], 
                num_heads=[6,12,24,48], 
                embed_dim=192, 
                num_classes=5, 
                model_size=model_size
            )
        elif model_size == "base":
            model = SwinTransformer(
                drop_rate=self.context.get_hparam("drop_rate"), 
                depths=[2, 2, 18, 2], 
                num_heads=[4, 8, 16, 32], 
                embed_dim=128, 
                num_classes=5, 
                model_size=model_size
            )
        elif model_size == "small":
            model = SwinTransformer(
                drop_rate=self.context.get_hparam("drop_rate"), 
                depths=[2, 2, 18, 2], 
                num_heads=[3, 6, 12, 24], 
                embed_dim=96, 
                num_classes=5, 
                model_size=model_size
            )
        else:
            raise ValueError(f"Invalid model size: {model_size}")

        model = torch.compile(model)
        return model

    def configure_optimizer(self, adamw_weight_decay_body, adamw_weight_decay_head, body_lr, head_lr):
        import inspect

        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Separate head parameters from the rest of the model
        head_params = [(n, p) for n, p in param_dict.items() if 'head' in n]
        backbone_params = [(n, p) for n, p in param_dict.items() if 'head' not in n]

        # Further divide into decay and no-decay groups
        decay_params = [p for n, p in backbone_params if p.dim() >= 2]
        no_decay_params = [p for n, p in backbone_params if p.dim() < 2]  # or 'bias' in n or 'norm' in n]

        head_decay_params = [p for n, p in head_params if p.dim() >= 2]
        head_no_decay_params = [p for n, p in head_params if p.dim() < 2]  # or 'bias' in n or 'norm' in n]

        # Group parameters for the optimizer
        optim_groups = [
            {"params": decay_params, "weight_decay": adamw_weight_decay_body, "lr": body_lr, 'lr_type': 0},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": body_lr, 'lr_type': 0},
            {"params": head_decay_params, "weight_decay": adamw_weight_decay_head, "lr": head_lr, 'lr_type': 1},
            {"params": head_no_decay_params, "weight_decay": 0.0, "lr": head_lr, 'lr_type': 1},
        ]

        # Check for fused AdamW
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        print(f"Using AdamW fused: {use_fused}")

        # Create optimizer
        optimizer = torch.optim.AdamW(optim_groups, fused=use_fused)
        return optimizer

    def build_training_data_loader(self) -> DataLoader:
        """Build and return the training DataLoader."""
        return DataLoader(
            get_train_dataset(root="/mnt/database/unaug_images/train"),  # Adjust path if needed
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.context.get_hparam("num_workers"),
            drop_last=True,
            pin_memory=True  # Optional: Speeds up data transfer to GPU
        )

    def build_validation_data_loader(self) -> DataLoader:
        """Build and return the validation DataLoader."""
        return DataLoader(
            get_val_dataset(root="/mnt/database/unaug_images/val"),  # Adjust path if needed
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            num_workers=self.context.get_hparam("num_workers"),
            drop_last=True,
            pin_memory=True  # Optional
        )

    def train_batch(self, batch, batch_idx, epoch_idx):
        data, labels = batch
        outputs = self.model(data)
        loss = self.loss_fn(outputs, labels)

        # Backward pass
        self.context.backward(loss)

        # Gradient clipping
        clip_value = self.context.get_hparam("gradient_clip_value")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

        # Optimizer step
        self.context.step_optimizer(self.optimizer)

        # Scheduler step
        self.scheduler.step()

        return {"loss": loss.item()}




    def evaluate_batch(self, batch):
        # Initialize metrics
        distance_list = [0, 0, 0, 0, 0]  # d=0, d=1, d=2, d=3, d=4
        binary_correct = 0
        total_datapoints = 0

        data, target_tensor = batch
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.loss_fn(outputs, target_tensor)

            # Get predictions
            top_index_tensor = torch.argmax(outputs, dim=1)


            # Calculate accuracy
            accuracy_tensor = (top_index_tensor == target_tensor).float()
            accuracy = accuracy_tensor.mean().item()
            total_datapoints += top_index_tensor.size(0)

            # Binary accuracy
            binary_predictions = (top_index_tensor > 0).float()  # DR present: 1, No DR: 0
            binary_targets = (target_tensor > 0).float()
            binary_correct += (binary_predictions == binary_targets).float().sum().item()

            # Distance metrics
            batch_distance_tensor = torch.abs(top_index_tensor - target_tensor)
            for i in range(5):
                distance_list[i] += (batch_distance_tensor == i).sum().item()


        # Normalize distances
        for i in range(5):
            distance_list[i] /= total_datapoints
        binary_accuracy = binary_correct / total_datapoints

        # Prepare metrics to return
        metrics = {
            "val_loss": loss.item(),
            "accuracy_severity": accuracy,  # Represents distance/0
            "accuracy_binary": binary_accuracy,
        }

        # Add distance metrics (exclude distance/0)
        for i in range(1, 5):
            metrics[f"distance/{i}"] = distance_list[i]


        return metrics
