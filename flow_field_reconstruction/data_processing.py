from pathlib import Path
import pandas as pd
import torch
from turbsim_file import TurbSimFile
import hydra
from omegaconf import DictConfig
from enum import Enum
from io import StringIO
from tqdm import tqdm
from typing import Dict, Any


class FileNames(Enum):
    TURBULENT_FIELD = 'TurbulentField.bts'
    AEROSENSE_OUTPUT = 'AerosenseOutput.out'


def save_tensors(ts_file: TurbSimFile, interim_path: Path, run_directory_name: str) -> None:
    """Save field.y, field.z, and t as PyTorch tensors."""
    torch.save(torch.tensor(ts_file.y), interim_path / f'{run_directory_name}_y_coordinates.pt')
    torch.save(torch.tensor(ts_file.z), interim_path / f'{run_directory_name}_z_coordinates.pt')
    torch.save(torch.tensor(ts_file.t), interim_path / f'{run_directory_name}_time_steps.pt')


def save_field_values(ts_file: TurbSimFile, interim_path: Path, run_directory_name: str) -> None:
    """Save field values as PyTorch tensors."""
    field_values_tensor = torch.tensor(ts_file["u"]).permute(1, 0, 2, 3)
    torch.save(field_values_tensor, interim_path / f'{run_directory_name}_field_values.pt')


def process_csv_row(csv_row: pd.DataFrame, output_data: pd.DataFrame, all_blade_sensors: Dict[str, Dict[str, Any]], other_sensors: Dict[str, Any]) -> None:
    """Process a row of the CSV output data and update the sensors dictionaries."""
    for key in output_data.keys():
        if key.endswith("1") or "B1" in key:
            new_key = key.replace("B1", "").replace("b1", "")
            if "Pitch" in key:
                new_key = new_key[3:-1]
            if new_key in all_blade_sensors["blade1"]:
                all_blade_sensors["blade1"][new_key].append(csv_row[key].values[0])
            else:
                all_blade_sensors["blade1"][new_key] = [csv_row[key].values[0]]
        elif key.endswith("2") or "B2" in key:
            new_key = key.replace("B2", "").replace("b2", "")
            if "Pitch" in key:
                new_key = new_key[3:-1]
            if new_key in all_blade_sensors["blade2"]:
                all_blade_sensors["blade2"][new_key].append(csv_row[key].values[0])
            else:
                all_blade_sensors["blade2"][new_key] = [csv_row[key].values[0]]
        elif key.endswith("3") or "B3" in key:
            new_key = key.replace("B3", "").replace("b3", "")
            if "Pitch" in key:
                new_key = new_key[3:-1]
            if new_key in all_blade_sensors["blade3"]:
                all_blade_sensors["blade3"][new_key].append(csv_row[key].values[0])
            else:
                all_blade_sensors["blade3"][new_key] = [csv_row[key].values[0]]
        else:
            if "RtAvg" not in key and not csv_row.empty:
                try:
                    if key in other_sensors:
                        other_sensors[key].append(csv_row[key].values[0])
                    else:
                        other_sensors[key] = [csv_row[key].values[0]]
                except:
                    print(f"Error processing key: {key}")


def save_sensors_to_csv(all_blade_sensors: Dict[str, Dict[str, Any]], other_sensors: Dict[str, Any], interim_path: Path, run_directory_name: str) -> None:
    """Save all_blade_sensors and other_sensors as CSV files."""
    for blade, sensors in all_blade_sensors.items():
        df = pd.DataFrame(sensors)
        df.to_csv(interim_path / f'{run_directory_name}_{blade}_sensors.csv', index=False)
    
    other_sensors_df = pd.DataFrame(other_sensors)
    other_sensors_df.to_csv(interim_path / f'{run_directory_name}_other_sensors.csv', index=False)


def process_run_directory(run_directory: Path, interim_path: Path, time_step: int) -> None:
    """Process a single run directory."""
    print(f"Processing run directory: {run_directory}")
    
    # Define the file paths using pathlib
    bts_file_path = run_directory / FileNames.TURBULENT_FIELD.value
    out_file_path = run_directory / FileNames.AEROSENSE_OUTPUT.value
    
    # Check if the .bts file exists
    if not bts_file_path.exists():
        print(f"File not found: {bts_file_path}")
        return
    
    # Check if the .out file exists
    if not out_file_path.exists():
        print(f"File not found: {out_file_path}")
        return
    
    # Read the turbulence file using the TurbSimFile class
    ts_file = TurbSimFile(bts_file_path)
    
    # Read the output data file as a normal file and skip the first 5 rows
    with open(out_file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()[6:]
        lines.pop(1)
    output_data = pd.read_csv(StringIO(''.join(lines)), sep='\s+')
    
    all_blade_sensors = {"blade1": {}, "blade2": {}, "blade3": {}}
    other_sensors = {}
    
    save_tensors(ts_file, interim_path, run_directory.name)
    save_field_values(ts_file, interim_path, run_directory.name)

    for i in tqdm(range(len(ts_file.t)), desc=f"Processing {run_directory.name}"):
        if i*time_step >= len(output_data):
            break
        csv_row = output_data.iloc[i*time_step:i*time_step+1, :]
        process_csv_row(csv_row, output_data, all_blade_sensors, other_sensors)
    
    save_sensors_to_csv(all_blade_sensors, other_sensors, interim_path, run_directory.name)


def read_turbulence_and_output_data(cfg: DictConfig) -> None:
    """Read turbulence and output data based on the cfg."""
    runs_path = Path(cfg.dataset_raw_dir)
    interim_path = Path(cfg.interim_path)
    interim_path.mkdir(parents=True, exist_ok=True)
    
    
    for run_directory in runs_path.iterdir():
        if run_directory.is_dir():
            process_run_directory(run_directory, interim_path, cfg.time_step)


@hydra.main(config_path="config/", config_name="data_processing")
def main(cfg: DictConfig) -> None:
    """Main function to read turbulence and output data."""
    read_turbulence_and_output_data(cfg)


if __name__ == "__main__":
    main()