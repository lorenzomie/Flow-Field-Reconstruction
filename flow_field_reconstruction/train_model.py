import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl

##############################################
# Dataset: Build subsequences of tokens from CSV
##############################################

class FlowSequenceDataset(Dataset):
    def __init__(self, data_file, subseq_length=1024, boundary_threshold=1e-3):
        """
        Loads flow data from a CSV file and returns fixed-length subsequences.
        
        The CSV is expected to have columns: y, z, v, w where:
          - y, z: spatial coordinates.
          - v, w: velocity components.
          
        For each token, we compute:
          - Boundary position encoding: normalize y and z (e.g., y from [-0.5,1.5] to [0,1],
            and z from [-0.5,0.5] to [0,1]).
          - Boundary value encoding: if the point is near the boundary (within boundary_threshold),
            we set the value to v, otherwise zero.
          
        The augmented input token is a 5-dimensional vector:
          [y, z, pos_enc_y, pos_enc_z, boundary_value_enc]
        The target output is a 2-dimensional vector: [v, w].
        
        To allow the transformer to work on sequences, we sort the data by spatial coordinates
        and then split the full dataset into sliding windows (subsequences) of fixed length.
        """
        # Load data from CSV (skip header row if present)
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        # Sort the data in spatial order (first by y then by z)
        data = data[np.lexsort((data[:, 1], data[:, 0]))]
        self.y = data[:, 0]
        self.z = data[:, 1]
        self.v = data[:, 2]
        self.w = data[:, 3]
        
        # Determine boundary points using the provided threshold.
        boundary_indices = (np.abs(self.y + 0.5) < boundary_threshold) | \
                           (np.abs(self.y - 1.5) < boundary_threshold) | \
                           (np.abs(self.z + 0.5) < boundary_threshold) | \
                           (np.abs(self.z - 0.5) < boundary_threshold)
        
        # Compute boundary position encodings (normalize coordinates to [0, 1]).
        pos_enc_y = (self.y + 0.5) / 2.0   # assuming domain y in [-0.5, 1.5]
        pos_enc_z = (self.z + 0.5) / 1.0   # assuming domain z in [-0.5, 0.5]
        
        # Compute boundary value encoding (using v component for boundary points).
        boundary_value_enc = np.zeros_like(self.v)
        boundary_value_enc[boundary_indices] = self.v[boundary_indices]
        
        # Build augmented input: shape (N, 5)
        X = np.stack([self.y, self.z, pos_enc_y, pos_enc_z, boundary_value_enc], axis=-1)
        # Build target output: shape (N, 2)
        Y = np.stack([self.v, self.w], axis=-1)
        
        # Store full sequences and subsequence length.
        self.X = X
        self.Y = Y
        self.subseq_length = subseq_length
        self.N = len(X)
        
        # Compute the number of subsequences (simple non-overlapping split).
        self.num_samples = self.N // self.subseq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a subsequence of tokens:
          - x: Tensor of shape (L, 5) where L is the subsequence length.
          - y: Tensor of shape (L, 2) corresponding to the target velocities.
        """
        start = idx * self.subseq_length
        end = start + self.subseq_length
        x = torch.tensor(self.X[start:end], dtype=torch.float32)
        y = torch.tensor(self.Y[start:end], dtype=torch.float32)
        return x, y

##############################################
# DataModule: Create train, validation, and test splits
##############################################

class FlowSequenceDataModule(pl.LightningDataModule):
    def __init__(self, data_file, subseq_length=1024, batch_size=1, train_split=0.7, val_split=0.15, test_split=0.15):
        """
        DataModule to load flow data and split it into training, validation, and test sets.
        
        Args:
            data_file (str): Path to the CSV file.
            subseq_length (int): Length of each subsequence.
            batch_size (int): Batch size for DataLoaders.
            train_split (float): Fraction of data for training.
            val_split (float): Fraction of data for validation.
            test_split (float): Fraction of data for testing.
        """
        super().__init__()
        self.data_file = data_file
        self.subseq_length = subseq_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def setup(self, stage=None):
        # Load the full sequence dataset.
        full_dataset = FlowSequenceDataset(self.data_file, self.subseq_length)
        num_samples = len(full_dataset)
        n_train = int(num_samples * self.train_split)
        n_val = int(num_samples * self.val_split)
        n_test = num_samples - n_train - n_val
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

##############################################
# Transformer Model with Cross Attention
##############################################

class FlowTransformer(pl.LightningModule):
    def __init__(self, d_model=64, nhead=4, num_layers=3, input_dim=5, target_dim=2):
        """
        Transformer model that uses cross attention to predict flow velocities.
        
        The model works as follows:
          - An input embedding layer maps the 5-dimensional token (augmented features) to a d_model-dimensional space.
          - A query embedding layer maps the spatial coordinates (first two components of the token) to d_model.
          - A stack of Transformer decoder layers (which include cross attention) allows each query token to
            attend to the full embedded input (memory).
          - A projection layer maps the resulting representation to the target output (velocity components).
        
        Args:
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer decoder layers.
            input_dim (int): Dimension of input tokens (default is 5).
            target_dim (int): Dimension of target output (default is 2 for v and w).
        """
        super().__init__()
        self.d_model = d_model
        
        # Embed the full augmented input (memory).
        self.input_embedding = nn.Linear(input_dim, d_model)
        # Embed the query tokens using the spatial coordinates (first two features).
        self.query_embedding = nn.Linear(2, d_model)
        
        # Create a stack of TransformerDecoderLayers.
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
            ) for _ in range(num_layers)
        ])
        # Final projection to the target velocity components.
        self.out_proj = nn.Linear(d_model, target_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (B, L, 5) where B is the batch size and L is the sequence length.
        
        Returns:
            Tensor: Predictions of shape (B, L, target_dim) corresponding to the velocity components.
        """
        # Embed the full input to form the memory.
        memory = self.input_embedding(x)  # (B, L, d_model)
        # Use the first two features (spatial coordinates) as query tokens and embed them.
        query = self.query_embedding(x[..., :2])  # (B, L, d_model)
        
        # Pass through the stack of Transformer decoder layers.
        out = query
        for layer in self.decoder_layers:
            # Each decoder layer uses cross attention where queries attend to memory.
            out = layer(tgt=out, memory=memory)
        
        # Project the attended features to the target output.
        out = self.out_proj(out)  # (B, L, target_dim)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, L, 5), y: (B, L, 2)
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


##############################################
# Example Usage
##############################################

if __name__ == '__main__':
    data_file = 'flow_data.csv'  # Path to your CSV file with columns: y, z, v, w
    subseq_length = 1024         # Adjust based on your data size
    batch_size = 1               # Each batch is one subsequence

    # Create the DataModule.
    data_module = FlowSequenceDataModule(data_file=data_file,
                                         subseq_length=subseq_length,
                                         batch_size=batch_size,
                                         train_split=0.7,
                                         val_split=0.15,
                                         test_split=0.15)
    
    # Instantiate the transformer model.
    model = FlowTransformer(d_model=64, nhead=4, num_layers=3, input_dim=5, target_dim=2)
    
    # Create a PyTorch Lightning trainer.
    trainer = pl.Trainer(max_epochs=500)
    
    # Train the model.
    trainer.fit(model, data_module)
    
    # (Optionally) Evaluate on the test set.
    trainer.test(model, datamodule=data_module)
