import pandas as pd
import numpy as np
from pathlib import Path
import re

class FungalDataLoader:
    """Load and preprocess fungal electrical data from various formats."""
    
    @staticmethod
    def detect_format(fp: Path) -> str:
        """Detect the format of the input file."""
        # Read first few lines
        with open(fp, 'r') as f:
            header = [next(f) for _ in range(5)]
        
        # Check for SigView format (single column of numbers)
        try:
            all(float(line.strip()) for line in header)
            return 'sigview'
        except ValueError:
            pass
            
        # Check for moisture logger format
        if any('moisture' in line.lower() or 'humidity' in line.lower() for line in header):
            return 'moisture'
            
        # Default to standard CSV format
        return 'standard'
    
    @staticmethod
    def convert_sigview(fp: Path) -> pd.DataFrame:
        """Convert SigView single-column format to standard format."""
        # Read raw values
        values = pd.read_csv(fp, header=None, names=['voltage'])
        
        # Create time column (assuming 1 second sampling)
        values['time'] = np.arange(len(values)) / 1.0
        
        return values[['time', 'voltage']]
    
    @staticmethod
    def convert_moisture(fp: Path) -> pd.DataFrame:
        """Convert moisture logger format to standard format."""
        # Skip metadata rows until we find the header
        skiprows = 0
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                if any(x in line.lower() for x in ['time', 'datetime', 'timestamp']):
                    skiprows = i
                    break
        
        # Read data with proper header row
        df = pd.read_csv(fp, skiprows=skiprows)
        
        # Find time and measurement columns
        time_col = next(col for col in df.columns if any(x in col.lower() for x in ['time', 'datetime', 'timestamp']))
        data_col = next(col for col in df.columns if any(x in col.lower() for x in ['moisture', 'humidity', 'measurement']))
        
        # Convert to standard format
        df = df[[time_col, data_col]]
        df.columns = ['time', 'voltage']
        
        return df
    
    @staticmethod
    def load_data(fp: Path) -> pd.DataFrame:
        """Load data from file, detecting and converting format as needed."""
        format_type = FungalDataLoader.detect_format(fp)
        
        if format_type == 'sigview':
            return FungalDataLoader.convert_sigview(fp)
        elif format_type == 'moisture':
            return FungalDataLoader.convert_moisture(fp)
        else:
            # Standard CSV format - find appropriate columns
            df = pd.read_csv(fp)
            
            # Find voltage/signal column
            voltage_col = None
            for col in df.columns:
                if any(k in col.lower() for k in ['mv', 'volt', 'v)', 'signal', 'potential']):
                    voltage_col = col
                    break
            if voltage_col is None:
                voltage_col = df.columns[1]  # Fallback to second column
                
            # Find or create time column
            if 'time' in df.columns:
                time_col = 'time'
            else:
                # Create time column based on sampling rate (default 1 Hz)
                df['time'] = np.arange(len(df)) / 1.0
                time_col = 'time'
            
            return df[[time_col, voltage_col]].rename(columns={voltage_col: 'voltage'}) 