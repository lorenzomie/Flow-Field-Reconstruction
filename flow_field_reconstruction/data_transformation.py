import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig
import hydra
import math
import numpy as np



def calculate_yz_positions(azimuth_list: list[float], distances: list[float]) -> Dict[str, np.ndarray]:
    """Calculate y-z positions for each blade based on the azimuth angle and distances."""
    positions = {"blade1": [], "blade2": [], "blade3": []}
    for azimuth_timestep in azimuth_list:
        blade1_pos = []
        blade2_pos = []
        blade3_pos = []
        for r in distances:
            y1 = r * math.cos(math.radians(azimuth_timestep))
            z1 = r * math.sin(math.radians(azimuth_timestep))
            y2 = r * math.cos(math.radians(azimuth_timestep + 120))
            z2 = r * math.sin(math.radians(azimuth_timestep + 120))
            y3 = r * math.cos(math.radians(azimuth_timestep + 240))
            z3 = r * math.sin(math.radians(azimuth_timestep + 240))
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


def process_sensor_data(interim_path: Path, processed_path: Path, run_names: Path, distances: list[float], grid) -> None:
    """Process sensor data from CSV files and save as PyTorch tensors."""
    for run_name in run_names:
        print(f"Processing run: {run_name}")
        other_sensors_file = interim_path / f'{run_name}_other_sensors.csv'
        
        # Read azimuth angle from other_sensors.csv
        other_sensors = pd.read_csv(other_sensors_file)
        azimuth = other_sensors['Azimuth'].values

        # Calculate y-z positions for each blade
        positions = calculate_yz_positions(azimuth, distances)

        blade1_pos = positions['blade1']
        blade2_pos = positions['blade2']
        blade3_pos = positions['blade3']

        # Read sensor data for each blade
        sensor_data = pd.DataFrame()
        for blade in ["blade1", "blade2", "blade3"]:
            blade_sensors_file = interim_path / f'{run_name}_{blade}_sensors.csv'
            blade_sensors = pd.read_csv(blade_sensors_file)

        # 1 to 9 are the sections of the blade
        alpha_keys = [f'N{i}Alpha' for i in range(1, 9)]
        vrel_keys = [f'N{i}Vrel' for i in range(1, 9)]
        dynp_keys = [f'N{i}DynP' for i in range(1, 9)]
        fn_keys = [f'N{i}Fn' for i in range(1, 9)]
        ft_keys = [f'N{i}Ft' for i in range(1, 9)]

        
        # Map sensor data to a 21x21 grid
        grid = map_sensors_to_grid(sensor_data, positions)

        # Save the grid as a PyTorch tensor
        torch.save(grid, processed_path / f'{run_name}_sensor_grid.pt')


@hydra.main(config_path="config/", config_name="data_processing")
def main(cfg: DictConfig) -> None:
    """Main function to process sensor data."""
    interim_path = Path(cfg.interim_path)
    processed_path = Path(cfg.processed_path)
    run_names = list(set([d.name[:7] for d in interim_path.iterdir() if d.name.startswith('run_') and d.name[4:7].isdigit()]))
    process_sensor_data(interim_path, processed_path, run_names, cfg.section_distances)


if __name__ == "__main__":
    main()