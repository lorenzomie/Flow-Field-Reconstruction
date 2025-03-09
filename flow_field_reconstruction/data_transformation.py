import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
import hydra
import math
import numpy as np



def calculate_yz_positions(azimuth: float, distances: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Calculate y-z positions for each blade based on the azimuth angle and distances."""
    positions = {"blade1": [], "blade2": [], "blade3": []}
    for _, row in distances.iterrows():
        r = row['distance']
        y1 = r * math.cos(math.radians(azimuth))
        z1 = r * math.sin(math.radians(azimuth))
        y2 = r * math.cos(math.radians(azimuth + 120))
        z2 = r * math.sin(math.radians(azimuth + 120))
        y3 = r * math.cos(math.radians(azimuth + 240))
        z3 = r * math.sin(math.radians(azimuth + 240))
        positions["blade1"].append((y1, z1))
        positions["blade2"].append((y2, z2))
        positions["blade3"].append((y3, z3))
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


def process_sensor_data(interim_path: Path, distances_file: Path, distances: list[float]) -> None:
    """Process sensor data from CSV files and save as PyTorch tensors."""
    for run_directory in interim_path.iterdir():
        if run_directory.is_dir():
            print(f"Processing run directory: {run_directory}")
            
            # Read azimuth angle from other_sensors.csv
            other_sensors_file = run_directory / f'{run_directory.name}_other_sensors.csv'
            other_sensors = pd.read_csv(other_sensors_file)
            azimuth = other_sensors['Azimuth'].iloc[0]

            # Calculate y-z positions for each blade
            positions = calculate_yz_positions(azimuth, distances)

            # Read sensor data for each blade
            sensor_data = pd.DataFrame()
            for blade in ["blade1", "blade2", "blade3"]:
                blade_sensors_file = run_directory / f'{run_directory.name}_{blade}_sensors.csv'
                blade_sensors = pd.read_csv(blade_sensors_file)
                sensor_data = pd.concat([sensor_data, blade_sensors], axis=1)

            # Map sensor data to a 21x21 grid
            grid = map_sensors_to_grid(sensor_data, positions)

            # Save the grid as a PyTorch tensor
            torch.save(grid, run_directory / f'{run_directory.name}_sensor_grid.pt')


@hydra.main(config_path="config/", config_name="data_processing")
def main(cfg: DictConfig) -> None:
    """Main function to process sensor data."""
    interim_path = Path(cfg.interim_path)
    processed_path = Path(cfg.processed_path)
    process_sensor_data(interim_path, processed_path, cfg.section_distances)


if __name__ == "__main__":
    main()