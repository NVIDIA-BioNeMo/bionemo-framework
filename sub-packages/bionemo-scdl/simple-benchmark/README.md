# SCDL Speedtest

## Overview

The SCDL-speedtest provides a single-script speedtest to measure the performance of BioNeMo Framework's Single Cell
Data Loader (SCDL) on your hardware to make sure it's performing as expected. It is designed to be easy to run,
to work with your AnnData files, and to produce a simple set of reported metrics representative of real performance
for applications using SCDL in a PyTorch DataLoader.

## Quick Start

### 0. Use a virtual environment

```bash
python -m venv bionemo_scdl_speedtest

source bionemo_scdl_speedtest/bin/activate
```

### 1. Install Dependencies

```bash
pip install torch pandas psutil tqdm bionemo-scdl
```

**For baseline comparison** (optional):

```bash
pip install anndata scipy
```

**Note**: If you have the BioNeMo source code, you can install bionemo-scdl locally:

```bash
cd /path/to/bionemo-framework
pip install -e sub-packages/bionemo-scdl/
```

### 2. Run Basic Benchmark

```bash
# Download example dataset and run a quick benchmark / smoke test.
python scdl_speedtest.py

# Benchmark your own AnnData dataset
python scdl_speedtest.py -i your_dataset.h5ad

# Export detailed CSV files
python scdl_speedtest.py --csv

# Export detailed JSON file
python scdl_speedtest.py --json results.json

# Run multiple iterations and average results for more stable benchmarks
python scdl_speedtest.py --num-runs 3
```

3. Deactivate your virtual environment to return to your original shell state

```bash
deactivate
```

## More Usage Examples

```bash
# Basic speedtest, using an automatically downloaded example dataset
python scdl_speedtest.py

# Test SCDL's expected performance on a specific AnnData dataset using sequential sampling
python scdl_speedtest.py -i my_data.h5ad -s sequential

# Generate CSV files for analysis
python scdl_speedtest.py --csv -o report.txt

# Export results to a JSON file
python scdl_speedtest.py --json my_benchmark_results.json

# Run 5 benchmark iterations and average the results for more reliable measurements
python scdl_speedtest.py --num-runs 5

# Run multiple iterations with baseline comparison
python scdl_speedtest.py -i my_data.h5ad --generate-baseline --num-runs 3

# Run multiple iterations and export CSV with averaged results
python scdl_speedtest.py --num-runs 5 --csv

# Run the speedtest with a custom batch size and runtime limit
python scdl_speedtest.py --batch-size 64 --max-time 60

# Baseline comparison (SCDL vs AnnData in backed mode with lazy loading)
python scdl_speedtest.py --generate-baseline

# Baseline comparison using a specific SCDL dataset path
python scdl_speedtest.py --generate-baseline -i my_data.h5ad --scdl-path /path/to/converted_scdl_data
```

## Command Line Options

| Option                  | Description                                                                                                          | Default                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `-i, --input`           | Dataset path (.h5ad, directory with .h5ad files, or scdl directory)                                                  | Auto-download example    |
| `-o, --output`          | Save report to file                                                                                                  | Print to screen (stdout) |
| `-s, --sampling-scheme` | Sampling method (shuffle/sequential/random)                                                                          | shuffle                  |
| `--batch-size`          | Batch size used in the PyTorch DataLoader                                                                            | 32                       |
| `--max-time`            | Max benchmark runtime (seconds). If the dataset is smaller                                                           | 30                       |
| `--warmup-time`         | Warmup period (seconds). This runs the dataloader before measurement to better reflect average expected performance. | 2                        |
| `--csv`                 | Export detailed CSV files                                                                                            | False                    |
| `--batch-size`          | Batch size used in the PyTorch DataLoader                                                                            | 64                       |
| `--warmup-time`         | Warmup period (seconds). This runs the dataloader before measurement to better reflect average expected performance. | 0                        |
| `--json`                | Export detailed JSON file to specified filename                                                                      | None                     |
| `--generate-baseline`   | Compare SCDL vs AnnData performance                                                                                  | False                    |
| `--scdl-path`           | Path to SCDL dataset (optional, only used with --generate-baseline)                                                  | None                     |
| `--num-epochs`          | The number of epochs (passes through the training dataset).                                                          | 1                        |
| `--num-runs`            | Number of benchmark runs to average (for more stable and reliable measurements)                                      | 1                        |
| `--use-X-not-raw`       | Set to use the .X, not the raw.X from an anndata file at conversion time                                             | None                     |

## Sample Output

```
============================================================
SCDL BENCHMARK REPORT
============================================================

Dataset: cellxgene_example_25k.h5ad
Method: SCDL
Sampling: shuffle
Epochs: 1

PERFORMANCE METRICS:
  Throughput:        20,098 samples/sec
  Instantiation:     0.066 seconds
  Avg Batch Time:    0.0016 seconds

MEMORY USAGE:
  Baseline:          446.6 MB
  Peak (Benchmark):  703.2 MB
  Dataset on Disk:   207.30 MB

DATA PROCESSED:
  Total Samples:     25,382 (25,382/epoch)
  Total Batches:     794 (794/epoch)
============================================================
SCDL version: 0.0.8
Anndata version: 0.11.4
```

## Baseline Comparison Output

When using `--generate-baseline`, you get a comprehensive comparison between SCDL and AnnData performance.

**Note:** The `--scdl-path` parameter is optional and can be used with `--generate-baseline` to specify an existing SCDL dataset path instead of converting from the input H5AD file. If not provided, the input H5AD file will be automatically converted to SCDL format for the comparison. This parameter is useful when you have already converted your data to SCDL format and want to benchmark against the same dataset without reconversion.

````
================================================================================
SCDL vs ANNDATA COMPARISON REPORT
================================================================================

Dataset: cellxgene_example_25k.h5ad
Sampling: shuffle

THROUGHPUT COMPARISON:
  SCDL:              22,668 samples/sec
  AnnData:           2,529 samples/sec
  Performance:       8.96x speedup with SCDL

MEMORY COMPARISON:
  SCDL Peak:         703.5 MB
  AnnData Peak:      568.8 MB
  Memory Efficiency: SCDL uses 1.24x more memory

DISK USAGE COMPARISON:
  SCDL Size:         0.20 GB
  AnnData Size:      0.14 GB
  Storage Efficiency: SCDL uses 1.43x more disk space

LOADING TIME COMPARISON:
  SCDL Conversion:   0.00 seconds (cached)
  AnnData Load:      0.25 seconds

SUMMARY:
  SCDL provides 9.0x throughput improvement
  SCDL uses 1.2x more memory
  SCDL disk usage: 0.20 GB
  AnnData disk usage: 0.14 GB
  SCDL uses 1.4x more disk space
================================================================================```
````

## CSV Export

When using `--csv`, the script generates:

- **`summary.csv`**: Overall benchmark metrics and configuration
- **`detailed_breakdown.csv`**: Per-epoch performance breakdown

Perfect for analysis in Excel, Python, R, or other data tools.

## JSON Export

When using `--json filename.json`, the script generates a comprehensive JSON file containing:

- **Metadata**: Export timestamp, number of results, and export version
- **Complete benchmark results**: All metrics from the `BenchmarkResult` dataclass
- **Derived metrics**: Calculated performance metrics (samples/sec, memory efficiency, etc.)
- **Per-epoch breakdowns**: Detailed performance data for each epoch (when available)

## Troubleshooting

### Dataset Issues

- **H5AD files**: Converted automatically to SCDL format (conversion time reported)
- **Large datasets**: Uses memory-mapped access for efficiency
- **Download failures**: Check internet connection and try again
- **Conversion caching**: H5AD files are converted once, then reused on subsequent runs

### Performance Tips

- **Faster throughput**: Use `--batch-size 64` or higher
- **Longer runs**: Increase `--max-time 120` for stable measurements
- **Memory profiling**: Use `--csv` to get detailed memory usage per epoch
- **Clearing the page cache**: With lazy loading, data may be stored in the page cache between runs. This is especially an issue with SCDL. Between runs, the page cache can be cleared with
  `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
- **Multiple runs for stability**: Use `--num-runs 3` or `--num-runs 5` to run multiple benchmark iterations and average the results. This reduces the impact of system variability, background processes, and cold cache effects, providing more reliable and representative performance measurements.

## Example Datasets

The script automatically downloads a 25K cell example dataset from CellxGene. For other datasets:

- **10X Genomics**: Convert .h5 files to .h5ad using `scanpy.read_10x_h5()`
- **AnnData files**: Use directly with `-i dataset.h5ad`
- **Large datasets**: Pre-convert to SCDL format for faster loading

### Tahoe 100M

The Tahoe 100M dataset (described in [Zhang _et al_. 2025](https://doi.org/10.1101/2025.02.20.639398)) contains data
from 1,100 small-molecule perturbations across 50,000 cancer cell lines, totaling 100 Million cells. This dataset was
used by [D'Ascenzo and Cultrera di Montesano 2025](https://github.com/Kidara/scDataset) to benchmark
dataloaders for single cell data.

To download the full Tahoe 100M dataset in AnnData format (1 file per plate, 14 total plates):

**Warning** This will trigger egress charges, which can be significant.

**Note** This dataset is 314 GB. The corresponding SCDL dataset after conversion is 1.1 TB,
so ensure that you have sufficient disk space if using the entire dataset.

**Note**: You will need to have installed the google cloud CLI to download this dataset.

```bash
gcloud storage cp -R gs://arc-ctc-tahoe100/2025-02-25/* .
```

This will download 19 total files (14 from the full set + 5 related to the tutorial).

To process this data, an option is to run `python scdl_speedtest.py --generate-baseline -i <path to h5ad>.`.
This will automatically convert the files to the SCDL format. Alternatively, with bionemo-scdl installed,
`convert_h5ad_to_scdl --data-path <path to h5ad> --save-path <SCDL path>`. This is a multi-hour process to run the
full conversion; however, running a single plate of the data should give you a good idea of expected SCDL performance
on your system. The following command will run the speedtest on the first plate, as downloaded above:

```bash
python scdl_speedtest.py --generate-baseline -i tahoe-100m/h5ad/plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad --warmup-time 30 --max-time 120 --use-X-not-raw
```

Alternatively, on the fully converted data:

```bash
python -m bionemo.scdl.simple_benchmark.scdl_speedtest --generate-baseline -i <path to Tahoe 100M in h5ad format> --scdl-path <path to Tahoe 100M in SCDL format> --warmup-time 30 --max-time 120
```

## Support

For support, please [file an issue in the BioNeMo Framework GitHub repository](https://github.com/NVIDIA-BioNeMo/bionemo-framework/issues).
This code will be updated and refactored once a general benchmarking framework is in place.
