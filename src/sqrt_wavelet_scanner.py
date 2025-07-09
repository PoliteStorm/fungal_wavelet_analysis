from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import os
import psutil  # NEW: for memory monitoring

# Local imports (relative)
from data_loader import FungalDataLoader
from fungal_networks.wavelet_analysis.sqrt_wavelet import SqrtWaveletTransform
from utils.split_large_csv import split_csv

MAX_FILE_BYTES = 20 * 1024 * 1024  # Reduced to 20 MB threshold
CHUNK_SIZE = 100_000  # Process 100k rows at a time

def get_memory_usage():
    """Return current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class SqrtWaveletScanner:
    """Batch-process fungal recordings using the √t wavelet transform."""

    def __init__(self,
                 data_root: str = "/home/kronos/AVALON/15061491",
                 output_root: str = "/home/kronos/AVALON/fungal_analysis/sqrt_wavelet_results",
                 file_extensions: List[str] = None):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.file_extensions = file_extensions if file_extensions is not None else [".csv"]
        self.loader = FungalDataLoader(data_dir=str(self.data_root))

    def _find_recordings(self) -> List[Path]:
        """Return all files under data_root with the desired extensions."""
        files: List[Path] = []
        for ext in self.file_extensions:
            files.extend(sorted(self.data_root.rglob(f"*{ext}")))
        return files

    def _select_data_column(self, df: pd.DataFrame) -> np.ndarray:
        """Heuristic to extract the most relevant signal column from a CSV."""
        # Prioritise columns containing voltage keywords
        for col in df.columns:
            if any(k in col.lower() for k in ["mv", "volt", "v)", "signal", "potential"]):
                data = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(data) > 10:
                    return data.values
        # Fallback: first numeric column after index/time
        for col in df.columns[1:]:
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(data) > 10:
                return data.values
        # Final fallback: first column
        return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values

    def process_chunks(self, fp: Path, sampling_rate: float):
        """Process a CSV file in chunks to manage memory usage."""
        print(f"  Initial memory usage: {get_memory_usage():.1f} MB")
        
        # First pass: count rows and identify target column
        sample_df = pd.read_csv(fp, nrows=10)
        total_rows = sum(1 for _ in open(fp)) - 1  # -1 for header
        print(f"  Total rows to process: {total_rows:,}")
        
        # Identify which column to use
        data_col = None
        for col in sample_df.columns:
            if any(k in col.lower() for k in ["mv", "volt", "v)", "signal", "potential"]):
                data_col = col
                break
        if data_col is None:
            data_col = sample_df.columns[1]  # Fallback to first non-index column
        
        # Process in chunks
        chunk_reader = pd.read_csv(fp, chunksize=CHUNK_SIZE)
        
        # Initialize aggregates
        all_coeffs = []
        all_mags = []
        all_phases = []
        
        for chunk_idx, chunk in enumerate(chunk_reader):
            print(f"  Processing chunk {chunk_idx + 1}/{(total_rows // CHUNK_SIZE) + 1}")
            signal_chunk = pd.to_numeric(chunk[data_col], errors="coerce").dropna().values
            
            if len(signal_chunk) < 10:  # Skip chunks with too little data
                continue
                
            # Process this chunk
            swt = SqrtWaveletTransform(sampling_rate=sampling_rate)
            coeffs, mag, phase = swt.analyze_signal(signal_chunk)
            
            all_coeffs.append(coeffs)
            all_mags.append(mag)
            all_phases.append(phase)
            
            if chunk_idx % 5 == 0:  # Report memory every 5 chunks
                print(f"  Memory usage: {get_memory_usage():.1f} MB")
                
            # Force garbage collection after each chunk
            del coeffs, mag, phase
            import gc
            gc.collect()
        
        # Aggregate results (average across chunks)
        print(f"  Aggregating results from {len(all_mags)} chunks")
        coeffs = np.mean(all_coeffs, axis=0)
        mag = np.mean(all_mags, axis=0)
        phase = np.mean(all_phases, axis=0)
        
        print(f"  Final memory usage: {get_memory_usage():.1f} MB")
        return coeffs, mag, phase

    def scan(self):
        recordings = self._find_recordings()
        print(f"Found {len(recordings)} files to process with sqrt-wavelet transform")

        for fp in recordings:
            try:
                print(f"Processing {fp.relative_to(self.data_root)} …")

                # Check file size first
                if fp.stat().st_size > MAX_FILE_BYTES:
                    print("  Large file detected; splitting into parts…")
                    split_csv(fp, rows_per_chunk=500_000, output_dir=fp.parent)
                    print("  Original file will be skipped; re-run scanner to process new parts")
                    continue

                # Process based on file type
                if fp.suffix.lower() == ".csv":
                    # Infer sampling rate: assume 1 Hz if "second" in file name else 10 Hz
                    sampling_rate = 1.0 if "second" in fp.stem.lower() else 10.0
                    coeffs, mag, phase = self.process_chunks(fp, sampling_rate)
                else:
                    print(f"Skipping unsupported format: {fp.suffix}")
                    continue

                # Save outputs
                out_dir = self.output_root / fp.stem
                out_dir.mkdir(parents=True, exist_ok=True)

                # Save magnitude heat-map (τ × k)
                plt.figure(figsize=(8, 6))
                sns.heatmap(mag, cmap="viridis", cbar_kws={"label": "|W(k,τ)|"})
                plt.xlabel("k index")
                plt.ylabel("τ index")
                plt.title("√t Wavelet Magnitude")
                plt.tight_layout()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(out_dir / f"sqrt_wavelet_magnitude_{timestamp}.png", dpi=300)
                plt.close()

                # Save raw arrays for downstream analysis
                np.save(out_dir / "coefficients.npy", coeffs)
                np.save(out_dir / "magnitude.npy", mag)
                np.save(out_dir / "phase.npy", phase)

                print("  ✓ Done")
                print(f"  Final memory usage: {get_memory_usage():.1f} MB")

            except Exception as e:
                print(f"  ✗ Error processing {fp.name}: {e}")


if __name__ == "__main__":
    scanner = SqrtWaveletScanner()
    scanner.scan() 