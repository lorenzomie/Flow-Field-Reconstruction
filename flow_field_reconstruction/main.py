from pathlib import Path
from turbsim_file import TurbSimFile

def main():
    file_path = Path('data/runs/run_001/TurbulentField.bts')
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
     
    ts_file = TurbSimFile(file_path)
    
    # Print some example information
    print("File read successfully!")
    print(f"z-coordinates: {ts_file.z}")
    print(f"y-coordinates: {ts_file.y}")
    print(f"time steps: {ts_file.t}")

if __name__ == "__main__":
    main()