"""
This script demonstrates how to train a Transformer model for flow field reconstruction using PyTorch Lightning.
The model uses cross attention to predict flow velocities from augmented input tokens.
"""
# Standard libraries
import math
import os
from omegaconf import DictConfig
from pathlib import Path
from enum import Enum
from typing import List, Dict, Tuple, Optional

# Plotting
import matplotlib.pyplot as plt
import numpy as np

# PyTorch Lightning
import pytorch_lightning as pl
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim

import hydra
from tqdm.notebook import tqdm

# MLflow
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping
mlflow.set_tracking_uri("http://127.0.0.1:8080")

##############################################
# Enum for File Types
##############################################

class FileType(Enum):
    ALx = "ALx"
    ALy = "ALy"
    Alpha = "Alpha"
    Dynp = "Dynp"
    Fn = "Fn"
    Ft = "Ft"
    MLx = "MLx"
    MLy = "MLy"
    Vrel = "Vrel"
    field_values = "field_values"

##############################################
# DataModule: Create train, validation, and test splits
##############################################

class FlowSequenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        """
        DataModule to load flow data from multiple files and split it into training, validation, and test sets.
        
        Args:
            cfg (DictConfig): Configuration object.
        """
        super().__init__()
        self.dataset_path: Path = Path(cfg.path)
        self.subseq_length: int = cfg.subseq_length
        self.batch_size: int = cfg.batch_size
        self.train_split: float = cfg.train_split
        self.val_split: float = cfg.val_split
        self.test_split: float = cfg.test_split

    def setup(self, stage: Optional[str] = None) -> None:
        # Find all files in the dataset path that match the pattern 'run_*'
        files: List[Path] = list(self.dataset_path.glob('run_*'))
        run_group: Dict[str, List[Path]] = self.split_files(files)
        
        # Initialize empty lists to store the data
        x_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []
        
        for group_key, group_files in run_group.items():
            dataset: Dict[str, torch.Tensor] = self.load_group_files(group_files)
        
            y: torch.Tensor = torch.tensor(dataset.pop("field_values"), dtype=torch.float32)  # (N, H, W, 3) velocity 3D
            x: torch.Tensor = torch.stack([torch.tensor(v, dtype=torch.float32) for v in dataset.values()]).permute(1, 2, 3, 0)  # (N, H, W, F) sensors data
            
            # Exclude last element of y and x for the last timestep padding
            x = x[:-1]
            y = y[:-1]

            # Reshape x to have the shape (N//5, 21, 21, 9)
            N: int = x.shape[0] // 5  # Number of timesteps of the sensor for 1 timestep of the field
            x = x[::5, :, :, :]
            y = y.permute(0, 2, 3, 1)  # (N, 21, 21, 3)

            # Append to the lists
            x_list.append(x)
            y_list.append(y)
        
        # Concatenate all tensors in the lists
        x_full: torch.Tensor = torch.cat(x_list, dim=0)
        y_full: torch.Tensor = torch.cat(y_list, dim=0)
        
        # Ensure the number of samples in x and y match
        assert x_full.shape[0] == y_full.shape[0], "Number of samples in x and y do not match"
        
        # Create the full dataset
        full_dataset: List[Tuple[torch.Tensor, torch.Tensor]] = [(x_full[i], y_full[i]) for i in range(x_full.shape[0])]
        
        num_samples: int = len(full_dataset)
        n_train: int = int(num_samples * self.train_split)
        n_val: int = int(num_samples * self.val_split)
        n_test: int = num_samples - n_train - n_val
        assert self.test_split == int(n_test / num_samples * 100) / 100
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    def split_files(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Split the list of files into training, validation, and test sets."""
        # Group files by the first 7 characters of their names
        file_groups: Dict[str, List[Path]] = {}
        for file in files:
            group_key: str = file.name[:7]
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(file)
        return file_groups

    def load_group_files(self, files: List[Path]) -> Dict[str, torch.Tensor]:
        """Load and process files in a group."""
        data: Dict[str, torch.Tensor] = {}
        for file in files:
            file_type: Optional[FileType] = self.get_file_type(file)
            if file_type:
                try:
                    file_data: torch.Tensor = torch.load(file)
                    if file_type.value in data:
                        print(f"Warning: Overwriting existing data for {file_type.value}")
                    data[file_type.value] = file_data
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        return data

    def get_file_type(self, file: Path) -> Optional[FileType]:
        """Determine the file type based on its name."""
        for file_type in FileType:
            if file_type.value in file.name:
                return file_type
        return None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int, width: int):
        """Positional Encoding for 2D inputs.

        Args:
            d_model: Hidden dimensionality of the input.
            height: Height of the 2D field.
            width: Width of the 2D field.
        """
        super().__init__()
        self.height: int = height
        self.width: int = width

        pe: torch.Tensor = torch.zeros(height, width, d_model)
        y_position: torch.Tensor = torch.arange(0, height, dtype=torch.float).unsqueeze(1).repeat(1, width)
        x_position: torch.Tensor = torch.arange(0, width, dtype=torch.float).unsqueeze(0).repeat(height, 1)

        div_term: torch.Tensor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, :, 0::2] = torch.sin(y_position.unsqueeze(-1) * div_term)  # (H, W, d_model/2)
        pe[:, :, 1::2] = torch.cos(x_position.unsqueeze(-1) * div_term)  # (H, W, d_model/2)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.

        Args:
            x: Input tensor of shape (B, H*W, d_model), where B is batch size, d_model is the embedding dimension, H is height, and W is width.
        
        Returns:
            Tensor: Input tensor with added positional encoding, shape (B, H*W, d_model).
        """
        B, H_W, d_model = x.shape
        # Add positional encoding to the input
        return x + self.pe.flatten(0, 1).unsqueeze(0)

class FlowTransformer(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        """
        Transformer model that uses cross attention to predict flow velocities.
        
        The model works as follows:
          - An input embedding layer maps the 5-dimensional token (augmented features) to a d_model-dimensional space.
          - A query embedding layer maps the spatial coordinates (first two components of the token) to d_model.
          - A stack of Transformer decoder layers (which include cross attention) allows each query token to
            attend to the full embedded input (memory).
          - A projection layer maps the resulting representation to the target output (velocity components).
        
        Args:
            cfg (DictConfig): Configuration object.
        """
        super().__init__()
        self.d_model: int = cfg.d_model
        self.nhead: int = cfg.nhead
        self.num_layers: int = cfg.num_layers
        self.feature_dim: int = cfg.feature_dim
        self.input_dim: int = cfg.height * cfg.width * cfg.feature_dim
        self.target_dim: int = cfg.target_dim
        self.dim_feedforward: int = cfg.dim_feedforward
        self.dropout: float = cfg.dropout
        self.height: int = cfg.height
        self.width: int = cfg.width
        self.alpha: float = cfg.alpha
        
        # Embed the full augmented input
        self.input_layer = nn.Conv2d(in_channels=self.feature_dim, out_channels=self.d_model, kernel_size=1)
        # Embed the query tokens using the spatial coordinates (first two features).
        self.query_embedding = nn.Linear(2, self.d_model)
        
        # Positional encoding for 2D field
        self.positional_encoding = PositionalEncoding2D(self.d_model, self.height, self.width)
        
        # Create a stack of TransformerDecoderLayers.
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, batch_first=True
            ) for _ in range(self.num_layers)
        ])
        # Final projection to the target velocity components.
        self.out_proj = nn.Linear(self.d_model, self.target_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (B, L, 5) where B is the batch size and L is the sequence length.
        
        Returns:
            Tensor: Predictions of shape (B, L, target_dim) corresponding to the velocity components.
        """
        B, H, W, F = x.shape  # B = batch size, H = height, W = width, F = feature dimension
        
        # Reshape x to (B, F, H, W) for convolution
        x_reshaped: torch.Tensor = x.permute(0, 3, 1, 2)  # (B, F, H, W)

        # Important: Memory act as residual in the class itself
        # Embed the full input to form the memory.
        memory: torch.Tensor = self.input_layer(x_reshaped)  # (B, d_model, H, W)
        memory = memory.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, d_model)

        # Apply positional encoding
        memory = self.positional_encoding(memory)  # (B, d_model, H, W) 

        spatial_coords: torch.Tensor = torch.stack(
            [torch.arange(0, H).repeat(W, 1).transpose(0, 1).flatten(),  # x-coordinates (height)
             torch.arange(0, W).repeat(H, 1).flatten()]  # y-coordinates (width)
        ).transpose(0, 1).float().to(x.device)  # (H*W, 2) for the query token


        # Embed the spatial coordinates to form the query tokens
        query: torch.Tensor = self.query_embedding(spatial_coords)  # (H * W, d_model)
  
        # Pass through the stack of Transformer decoder layers
        out: torch.Tensor = query.unsqueeze(0).repeat(B, 1, 1)  # Ensure the query shape matches batch size
        for layer in self.decoder_layers:
            out = layer(tgt=out, memory=memory)
        
        # Project the attended features to the target output (e.g., velocity components)
        out = self.out_proj(out)  # (B, L, target_dim)

        # Reshape back to (B, target_dim, H, W) for image-like output
        return out.view(B, H, W, self.target_dim)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        similarity_loss = cosine_similarity_loss_2d(y, y_hat)
        loss = mse_loss + self.alpha * similarity_loss
        self.log("train_loss", loss)
        self.log("mse_loss", mse_loss)
        self.log("similarity_loss", similarity_loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        similarity_loss = cosine_similarity_loss_2d(y, y_hat)
        loss = mse_loss + self.alpha * similarity_loss
        self.log("val_loss", loss)  # Change log key to 'val_loss'
        self.log("mse_loss", mse_loss)
        self.log("similarity_loss", similarity_loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        similarity_loss = cosine_similarity_loss_2d(y, y_hat)
        loss = mse_loss + self.alpha * similarity_loss
        self.log("test_loss", loss)
        self.log("mse_loss", mse_loss)
        self.log("similarity_loss", similarity_loss)
        return loss

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[Dict[str, object]]]:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

def cosine_similarity_loss_2d(y_true, y_pred):
    """
    Computes the cosine similarity between two 2D fields of vectors (e.g., velocity or direction).
    
    Args:
        y_true (torch.Tensor): True tensor of shape (B, H, W, d)
        y_pred (torch.Tensor): Predicted tensor of shape (B, H, W, d)
        
    Returns:
        torch.Tensor: Cosine similarity tensor of shape (B, H, W)
    """
    # Flatten the fields into (B * H * W, d)
    y_true_flat = y_true.view(-1, y_true.shape[-1])  # (B * H * W, d)
    y_pred_flat = y_pred.view(-1, y_pred.shape[-1])  # (B * H * W, d)

    # Compute the cosine similarity between the corresponding vectors
    cos_sim = F.cosine_similarity(y_true_flat, y_pred_flat, dim=-1)  # (B * H * W)
    
    # Reshape the result back to (B, H, W)
    cos_sim = cos_sim.view(y_true.shape[0], y_true.shape[1], y_true.shape[2])  # (B, H, W)
    
    cos_sim_loss = 1 - cos_sim  # Shape: (B, H, W)

    # Average the loss over all positions and batch elements
    cos_sim_loss = cos_sim_loss.mean()  # Scalar

    return cos_sim_loss

##############################################
# Example Usage
##############################################

@hydra.main(config_path="config", config_name="train_config.yaml")
def main(cfg: DictConfig) -> None:

    mlflow.pytorch.autolog()

    # Should get the hydra runtime directory
    mlflow_uri: str = f"file:{Path(os.getcwd()).parent.parent.parent}/mlruns/"
    print(f"MLFlow URI: {mlflow_uri}")

    # Create an MLFlow logger
    mlflow_logger = MLFlowLogger(experiment_name="flow_field_reconstruction", tracking_uri=mlflow_uri)

    # Create the DataModule.
    data_module = FlowSequenceDataModule(cfg.dataset)    
    
    # Instantiate the transformer model.
    model = FlowTransformer(cfg.model)
    
    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,        # Verbosity mode
        mode='min',          # Mode for the monitored metric ('min' or 'max')
        min_delta=0.02      # Minimum change in the monitored metric to qualify as an improvement
    )

    # Create a PyTorch Lightning trainer.
    trainer = pl.Trainer(
        max_epochs=cfg.model.max_epochs, 
        logger=mlflow_logger,
        callbacks=[early_stopping]  # Add the early stopping callback
    )
    
    # Train the model.
    trainer.fit(model, data_module)
    
    # Evaluate on the test set.
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()
