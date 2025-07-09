from pathlib import Path
import pandas as pd
import argparse


def split_csv(input_path: Path, rows_per_chunk: int = 500_000, output_dir: Path = None):
    """Split *input_path* into multiple smaller CSVs, each with *rows_per_chunk* lines.

    The header row is preserved in every chunk.  Output files are stored in
    *output_dir* (defaults to same directory) and named
    `<stem>_part<N>.csv`.
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Splitting {input_path} → {output_dir} (≈{rows_per_chunk} rows per chunk)…")

    reader = pd.read_csv(input_path, chunksize=rows_per_chunk)
    for idx, chunk in enumerate(reader, start=1):
        out_file = output_dir / f"{input_path.stem}_part{idx}{input_path.suffix}"
        chunk.to_csv(out_file, index=False)
        print(f"  wrote {out_file} ({len(chunk)} rows)")

    print("✓ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large CSV into smaller parts.")
    parser.add_argument("input_csv", type=str, help="Path to the large CSV file")
    parser.add_argument("--rows", type=int, default=500_000, help="Rows per chunk (default 500k)")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default same as input)")
    args = parser.parse_args()

    split_csv(Path(args.input_csv), rows_per_chunk=args.rows, output_dir=Path(args.out) if args.out else None) 