import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
import hydra
import math
import numpy as np
from collections import Counter
from tqdm import tqdm



def calculate_yz_positions(azimuth_list: list[float], distances: list[float], height: float) -> Dict[str, np.ndarray]:
    """Calculate y-z positions for each blade based on the azimuth angle and distances."""
    positions = {"blade1": [], "blade2": [], "blade3": []}
    for azimuth_timestep in azimuth_list:
        blade1_pos = []
        blade2_pos = []
        blade3_pos = []
        for r in distances:
            y1 = r * math.cos(math.radians(azimuth_timestep))
            z1 = r * math.sin(math.radians(azimuth_timestep)) + height
            y2 = r * math.cos(math.radians(azimuth_timestep + 120))
            z2 = r * math.sin(math.radians(azimuth_timestep + 120)) + height
            y3 = r * math.cos(math.radians(azimuth_timestep + 240))
            z3 = r * math.sin(math.radians(azimuth_timestep + 240)) + height
            blade1_pos.append((y1, z1))
            blade2_pos.append((y2, z2))
            blade3_pos.append((y3, z3))
        positions["blade1"].append(blade1_pos)
        positions["blade2"].append(blade2_pos)
        positions["blade3"].append(blade3_pos)
    return positions


def map_sensors_to_grid(sensor_data: pd.DataFrame, positions: Dict[str, np.ndarray], grid_size: int = 21) -> torch.Tensor:
    """Map sensor data to a 21x21 grid based on their y-z positions."""
    grid = torch.zeros((grid_size, grid_size, 2))  # 2 channels for v and w
    y_min, y_max = -10.0, 10.0  # Assuming y ranges from -10 to 10
    z_min, z_max = -10.0, 10.0  # Assuming z ranges from -10 to 10
    y_step = (y_max - y_min) / (grid_size - 1)
    z_step = (z_max - z_min) / (grid_size - 1)

    for blade, pos_list in positions.items():
        for i, (y, z) in enumerate(pos_list):
            grid_y = int((y - y_min) / y_step)
            grid_z = int((z - z_min) / z_step)
            grid[grid_y, grid_z, 0] = sensor_data[f'{blade}_v'].iloc[i]
            grid[grid_y, grid_z, 1] = sensor_data[f'{blade}_w'].iloc[i]

    return grid


def process_sensor_data(interim_path: Path, processed_path: Path, run_names: Path, distances: list[float], tower_heigth: float) -> None:
    """Process sensor data from CSV files and save as PyTorch tensors."""
    grid_y = torch.load(processed_path / 'y_coordinates.pt').numpy()
    grid_z = torch.load(processed_path / 'z_coordinates.pt').numpy()
    threshold = np.sqrt(np.max(np.diff(grid_y))**2 + np.max(np.diff(grid_z))**2) / 2 - 0.5 # hardcoded
    for run_name in tqdm(run_names, desc="Processing runs"):
        other_sensors_file = interim_path / f'{run_name}_other_sensors.csv'
        
        # Read azimuth angle from other_sensors.csv
        other_sensors = pd.read_csv(other_sensors_file)
        azimuth = other_sensors['Azimuth'].values

        # Calculate y-z positions for each blade
        positions = calculate_yz_positions(azimuth, distances, tower_heigth)

        blade1_pos = np.array(positions['blade1'])
        blade2_pos = np.array(positions['blade2'])
        blade3_pos = np.array(positions['blade3'])

        blade_masks = {'blade1': np.array([generate_blade_mask(blade1_pos, grid_y, grid_z, threshold=threshold) for blade1_pos in blade1_pos]), 
                 'blade2': np.array([generate_blade_mask(blade2_pos, grid_y, grid_z, threshold=threshold) for blade2_pos in blade2_pos]), 
                 'blade3': np.array([generate_blade_mask(blade3_pos, grid_y, grid_z, threshold=threshold) for blade3_pos in blade3_pos])}

        keys = {
            'Alpha': [f'N{i}Alpha' for i in range(1, 10)],
            'Vrel': [f'N{i}VRel' for i in range(1, 10)],
            'Dynp': [f'N{i}DynP' for i in range(1, 10)],
            'Fn': [f'N{i}Fn' for i in range(1, 10)],
            'Ft': [f'N{i}Ft' for i in range(1, 10)],
            'ALx': [f'Spn{i}ALx' for i in range(1, 10)],
            'ALy': [f'Spn{i}ALy' for i in range(1, 10)],
            'MLx': [f'Spn{i}MLx' for i in range(1, 10)],
            'MLy': [f'Spn{i}MLy' for i in range(1, 10)],
        }

        combined_data = {key: combine_sensor_data_with_mask(blade_masks, interim_path, run_name, value) for key, value in keys.items()}

        # Save the combined data as PyTorch tensors
        for key, data in combined_data.items():
            torch.save(data, processed_path / f'{run_name}_combined_{key}.pt')


def combine_sensor_data_with_mask(blade_masks: Dict, interim_path, run_name, keys):

    blade1_sensors = pd.read_csv(interim_path / f'{run_name}_blade1_sensors.csv')
    blade2_sensors = pd.read_csv(interim_path / f'{run_name}_blade2_sensors.csv')
    blade3_sensors = pd.read_csv(interim_path / f'{run_name}_blade3_sensors.csv')
        
    combined_data = blade1_sensors[keys].to_numpy()[:, :, None, None] * blade_masks["blade1"] + blade2_sensors[keys].to_numpy()[:, :, None, None] * blade_masks["blade2"] + blade3_sensors[keys].to_numpy()[:, :, None, None] * blade_masks["blade3"]
    
    # Reconstruct the boundary condition of the field
    assert combined_data.shape[1] == 9

    combined_data = np.sum(combined_data, axis=1)
    return combined_data

def generate_blade_mask(blade_pos, grid_y, grid_z, threshold=7.25):
    """
    Generate a binary mask (0 and 1) on a 21x21 grid based on the closest blade positions.

    Parameters:
    - blade_pos: np.array of shape (9,2) -> Coordinates of blade 1.
    - threshold: float -> Distance threshold to assign mask value (default: 0.1).

    Returns:
    - mask: np.array of shape (grid_size, grid_size) -> Binary mask where 1 represents proximity to blade positions.
    """
    mask_num = blade_pos.shape[0] # 9 Section 
    assert mask_num == 9, "Only 9 sections are accounted"
    grid_y_mesh, grid_z_mesh = np.meshgrid(grid_y, grid_z, indexing='ij')
    grid_points = np.stack([grid_y_mesh.ravel(), grid_z_mesh.ravel()], axis=1)  # Shape (N, 2)
    
    # Compute distances efficiently using broadcasting
    distances = np.linalg.norm(grid_points[:, None, :] - blade_pos[None, :, :], axis=2)  # Shape (N, num_blades)
    
    # Get the minimum distance for each grid point
    min_distances_idx = np.argmin(distances, axis=0)
    min_distances_idx = increment_duplicates(min_distances_idx) 

    masks = np.zeros((mask_num, *(grid_y_mesh).shape), dtype=int)  # Initialize all masks as zero

    # Convert 1D indices to 2D coordinates
    rows, cols = np.unravel_index(min_distances_idx, (grid_y_mesh).shape)

    # Assign 1 at the corresponding locations in each mask
    masks[np.arange(mask_num), rows, cols] = 1  
    return masks

def increment_duplicates(arr):
    counts = Counter(arr)
    seen = set()
    result = []
    
    for value in arr:
        if value in seen:
            value += 1
        seen.add(value)
        result.append(value)
    
    return np.array(result)

@hydra.main(config_path="config/", config_name="data_processing")
def main(cfg: DictConfig) -> None:
    """Main function to process sensor data."""
    interim_path = Path(cfg.interim_path)
    processed_path = Path(cfg.processed_path)
    run_names = list(set([d.name[:7] for d in interim_path.iterdir() if d.name.startswith('run_') and d.name[4:7].isdigit()]))
    # HP Height of the hub is the height of the tower
    tower_heigth = cfg.hub_height
    process_sensor_data(interim_path, processed_path, run_names, cfg.section_distances, tower_heigth = tower_heigth)


if __name__ == "__main__":
    main()