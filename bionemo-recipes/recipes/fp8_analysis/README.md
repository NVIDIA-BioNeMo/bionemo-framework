# FP8 Training Analyzer - User Guide

A model-agnostic tool for analyzing FP8 quantization logs and visualizing gradient underflows during training.

______________________________________________________________________

## 🎯 What Does This Tool Do?

The FP8 Analyzer helps you **diagnose training issues** caused by FP8 quantization by:

1. **Parsing training logs** - Extracts FP8 metrics from your training runs
2. **Auto-detecting model architecture** - Identifies encoder layers, head layers, and embeddings
3. **Generating publication-quality heatmaps** - Visualizes gradient underflows across all model components over time
4. **Exporting structured data** - Saves metrics to CSV for further analysis

### Key Features

✅ **Model-agnostic** - Works with any transformer architecture (ESM, BERT, GPT, T5, etc.)
✅ **Automatic component detection** - No configuration needed
✅ **Beautiful visualizations** - 600 DPI publication-ready heatmaps
✅ **Easy comparison** - Run on multiple experiments with suffixes

## How to use it

Before using this tool, you must first gather FP8 statistics during training. We currently support the following models
and training scripts.

| Model  | DDP | FSDP2 | MFSDP |
| ------ | --- | ----- | ----- |
| ESM2   | ✓   | ✓     | ✗     |
| LLAMA3 | ✓   | ✓     | ✗     |

To gather FP8 statistics for analysis, refer to the model-specific documentation (e.g., [ESM2 quantized training](../esm2_native_te/README.md#quantized-training-fp8-mxfp8-nvfp4)) or add these arguments to your training command:

```python
python train_fsdp2.py \
fp8_stats_config.enabled=True # whether to log stats or not
fp8_stats_config.fp8_log_dir=./logs/fp8_stats_logs_dummy # where to store the logs
fp8_stats_config.fp8_stats_file=./fp8_stats.yaml # specifies what stats you want to run. Currently this is saved in this yaml file.
fp8_config.enabled=True # set this to use FP8 otherwise stats logging won't work
```

Once the run is completed. The `fp8_stats_config.fp8_log_dir` should have several directories under it. It should look like this

```bash
└── rank_0
    ├── nvdlfw_inspect_logs
    │   └── nvdlfw_inspect_globalrank-0.log
    └── nvdlfw_inspect_statistics_logs
        └── nvdlfw_inspect_globalrank-0.log
```

Here we can see that there are directories for each rank. This is intended in case one wants to do a rank-by-rank analysis.
As we can see, there are `inspect_logs` and `inspect_statistics_logs`. The `inspect_logs` will tell you what layer names are being tracked as well as what tensor values are being logged.
The `inspect_statistics_logs` holds the actual stats for the runs, which should have a value for every tracked tensor at iterations specified by the `freq` parameter in the log config file (specified by `fp8_stats_config.fp8_stats_file`).

______________________________________________________________________

## 📊 Sample Output

### Example: Full FP8 Run (Encoder + Head in FP8)

**Command:**

```bash
python3 analyze_and_create_heatmap.py fp8logswithhead
```

**Result:**

Running the command writes a high-resolution heatmap to
`heatmap_visualization/heatmap_highres_fp8head.png`. The generated PNG is not
checked into the repository, so this guide describes the expected output
instead of embedding the file directly.

**What you see:**

- **35 components**: 33 encoder layers + 2 head layers (Dense, Decoder)
- **Red/orange bands**: Critical underflows in early layers (2-5) and late layer (33)
- **Head layers affected**: Both Dense (2.3%) and Decoder (2.5%) show underflows
- **White separator line**: Divides encoder layers from head layers
- **U-shape pattern**: Middle layers (7-28) are fine, but edges suffer
- **Max underflow**: 5.89% at Layer 33
- **Yellow boxes**: Highlight the 5 worst components

**Interpretation:**
This is a **problematic run** where FP8 quantization causes significant gradient underflows throughout the model. The head being in FP8 amplifies problems in the encoder. Early layers (2-5) suffer from vanishing gradients, while late layers (32-33) and head layers receive noisy gradients from FP8 quantization.

______________________________________________________________________

## 🚀 Quick Start

### 1. Run the Analyzer

```bash
python3 analyze_and_create_heatmap.py <log_directory>
```

### 2. View Output

The script generates two files:

```
analysis_output/csv_data/rank_0_metrics.csv
heatmap_visualization/heatmap_highres.png
```

Open the PNG to see your heatmap!

______________________________________________________________________

## 📖 Understanding the Heatmap

### Color Scale

| Color         | Underflow % | Meaning                                            |
| ------------- | ----------- | -------------------------------------------------- |
| 🟢 **Green**  | < 0.5%      | ✅ **Acceptable** - Normal quantization noise      |
| 🟡 **Yellow** | 0.5-2%      | ⚠️ **Warning** - Monitor but not critical          |
| 🟠 **Orange** | 2-4%        | 🔶 **Critical** - Significant learning signal loss |
| 🔴 **Red**    | > 4%        | ❌ **Severe** - Major training instability risk    |

### Visual Elements

1. **Yellow boxes** - Highlight the 5 worst components (>2% underflows)
2. **White separator lines** - Divide component groups (Encoder | Head | Embedding)
3. **Cyan vertical line** - Marks iteration 3000 (common divergence point)
4. **Side labels** - Show component groups (ENCODER, HEAD)
5. **Summary box** (top-right) - Key statistics

### Interpreting Patterns

#### ✅ Good Pattern

```
Most layers: Green
Few yellow spots: Acceptable
Max < 2%: Safe to continue training
```

#### ⚠️ Warning Pattern

```
Some layers: Orange (2-4%)
Isolated to 1-3 layers: Monitor closely
Max < 4%: Consider adjusting FP8 settings
```

#### ❌ Bad Pattern

```
Multiple layers: Red (>4%)
U-shape (early + late): Gradient flow issues
Max > 5%: High risk of divergence
```

______________________________________________________________________

## 🔍 Common Scenarios

### Scenario 1: U-Shape Pattern (Early + Late Layers)

**What it looks like:**

- Layers 1-5: Red/Orange
- Layers 6-28: Green
- Layers 29-33: Red/Orange
- Head: Red/Orange

**Why it happens:**

- **Early layers**: Far from loss, gradients shrink through backprop (vanishing gradient)
- **Late layers**: Close to loss but receive noisy gradients from head
- **Middle layers**: Goldilocks zone - far enough to have stable gradients, close enough to receive clean signal

**Solution:**

```python
# Keep problematic layers in higher precision
fp8_skip_layers = [
    "layers.1",
    "layers.2",
    "layers.3",  # Early layers
    "layers.31",
    "layers.32",
    "layers.33",  # Late layers
    "lm_head.dense",
    "lm_head.decoder",  # Head
]
```

### Scenario 2: Head-Only Problem

**What it looks like:**

- Encoder layers: Mostly green
- Head (Dense/Decoder): Red/Orange

**Why it happens:**

- Head has small vocabulary weight matrix with large dynamic range
- Gradients from cross-entropy loss can be very small or very large
- FP8 struggles with this high dynamic range

**Solution:**

```python
# Keep head in BF16
fp8_enabled = True
fp8_skip_layers = ["lm_head.dense", "lm_head.decoder"]
```

**Expected improvement:** 34-57% reduction in encoder underflows (validated!)

______________________________________________________________________

## 📐 Reading the Log Output

When you run the script, you'll see:

```
INFO - ================================================================================
INFO - MODEL-AGNOSTIC FP8 LOG ANALYZER & HEATMAP GENERATOR
INFO - ================================================================================
INFO - Log directory: fp8logswithhead
INFO - Output suffix: '_fp8head' (if provided)
INFO - ================================================================================

INFO - ================================================================================
INFO - PARSING MODEL ARCHITECTURE
INFO - ================================================================================
INFO - Metadata: fp8logswithhead/rank_0/nvdlfw_inspect_logs/nvdlfw_inspect_globalrank-0.log
INFO - Found 373 layer names

INFO - Model Structure:
INFO -   Encoder layers: 33
INFO -     Range: Layer 1 to 33
INFO -   Head layers: 3
INFO -     - model.lm_head
INFO -     - model.lm_head.dense
INFO -     - model.lm_head.decoder

INFO - ================================================================================
INFO - PARSING LOG FILE
INFO - ================================================================================
INFO - File: fp8logswithhead/.../nvdlfw_inspect_globalrank-0.log
INFO - Processed 500,000 lines...
INFO - Total lines: 6,013,920
INFO - Metrics extracted: 6,013,920
INFO - Iteration range: 0 to 7369

INFO - ================================================================================
INFO - AUTO-DETECTING COMPONENTS
INFO - ================================================================================
INFO - Found 68 gradient underflow metrics

INFO - Component Summary:
INFO -   Encoder: 33 components
INFO -   Head: 2 components
INFO -     - Decoder
INFO -     - Dense

INFO - ================================================================================
INFO - CREATING HEATMAP
INFO - ================================================================================
INFO - Components: 35
INFO - Data points: 257,950
INFO - Heatmap dimensions: 35 components × 121 time points

INFO - ✨ Saved heatmap: heatmap_visualization/heatmap_highres_fp8head.png
INFO -    Max underflow: 5.89%
INFO -    Critical components (>2%): 5

INFO - ================================================================================
INFO - ✅ COMPLETE
INFO - ================================================================================
```

**Key metrics to watch:**

- **Max underflow**: Should be < 2% ideally, < 4% acceptable
- **Critical components**: Fewer is better
- **Iteration range**: Ensure you have enough data

______________________________________________________________________

## 🔬 Advanced Analysis

### Comparing Multiple Runs

To compare different experiments, run the analyzer in separate directories or rename the output files after each run:

```bash
# Run 1: Analyze and save results
python3 analyze_and_create_heatmap.py logs_fp8_full
mv heatmap_visualization/heatmap_highres.png heatmap_visualization/run1_fp8.png
mv analysis_output/csv_data/rank_0_metrics.csv analysis_output/csv_data/run1_fp8.csv

# Run 2: Analyze next experiment
python3 analyze_and_create_heatmap.py logs_bf16_head
mv heatmap_visualization/heatmap_highres.png heatmap_visualization/run2_bf16.png
mv analysis_output/csv_data/rank_0_metrics.csv analysis_output/csv_data/run2_bf16.csv
```

Then compare the heatmaps side-by-side!

### Extracting Specific Metrics

The CSV output contains all metrics:

```python
import pandas as pd

# Load data
df = pd.read_csv("analysis_output/csv_data/rank_0_metrics.csv")

# Get Layer 33 underflows over time
layer33 = df[
    df["metric_name"]
    == "model.esm.encoder.layers.33.self_attention.layernorm_qkv_gradient_underflows%"
]

# Plot
import matplotlib.pyplot as plt

plt.plot(layer33["iteration"], layer33["value"])
plt.xlabel("Iteration")
plt.ylabel("Gradient Underflow %")
plt.title("Layer 33 Gradient Underflows")
plt.show()
```

______________________________________________________________________

## ❓ FAQ

### Q: What if my model has a different architecture?

**A:** The script auto-detects layers! It works with:

- ESM: `model.esm.encoder.layers.N`
- BERT: `model.encoder.layer.N`
- GPT: `model.transformer.layers.N`
- Custom: Any pattern with `.layers.N.` or `.layer.N.`

### Q: Why are some layers missing in my heatmap?

**A:** If you used `fp8_skip_layers`, those layers won't be in FP8 and won't have underflow metrics logged. This is expected!

### Q: What's a "good" underflow percentage?

**A:**

- **< 0.5%**: Excellent
- **0.5-1%**: Good
- **1-2%**: Acceptable
- **2-4%**: Concerning
- **> 4%**: Critical - action needed

### Q: Can I change the color scale?

**A:** Yes! Edit the `max_val` variable in the `create_heatmap` function in `analyze_and_create_heatmap.py`:

```python
max_val = min(6, pivot_sample.values.max())  # Change 6 to your max
```

### Q: How do I export to PDF?

**A:** Use ImageMagick or Preview.app:

```bash
# macOS
open heatmap_visualization/heatmap_highres.png
# File → Export as PDF

# Linux
convert heatmap_visualization/heatmap_highres.png output.pdf
```

______________________________________________________________________

______________________________________________________________________

## 🎓 Key Takeaways

1. **Gradient underflows % shows how much learning signal is lost** - Keep it under 2%
2. **U-shape pattern indicates vanishing/noisy gradients** - Fix by keeping problematic layers in BF16
3. **Head precision matters** - BF16 head reduces encoder underflows by 34-57%
4. **Compare runs visually** - Side-by-side heatmaps quickly show improvements
5. **Early detection is key** - Run this analyzer frequently during training

______________________________________________________________________

**Need help?** Check the example heatmaps above or refer to the FAQ section!

*Generated: January 13, 2026*
