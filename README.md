# Physics-Informed GNN Weather Downscaling

This project builds a physics-informed graph neural network for weather downscaling over Nepal. The core idea is to use one static terrain graph and many time-varying atmospheric feature tensors.

The graph represents fixed locations over Nepal. Each node stores terrain information such as elevation, slope, and aspect. Edges connect nearby nodes using k-nearest neighbors and include spatial and terrain relationship attributes. ERA5 weather variables are then interpolated onto those fixed graph nodes for each timestep.

## Project Status

Implemented:

- SRTM DEM download and mosaic for Nepal
- Static terrain graph construction
- DEM preprocessing utilities
- ERA5 predictor download script
- ERA5-Land target download and preprocessing scripts
- Monthly target tensor builder
- ERA5 preprocessing into node-aligned dynamic tensors
- Dataset class that combines static and dynamic features at runtime
- PI-GNN model layers, model definition, and loss functions
- Baseline training scripts for interpolation, MLP, and XGBoost
- GNN training, evaluation, checkpointing, and sweep scripts
- Inference and quantile-mapping utilities
- Graph visualization script

## Repository Layout

```text
configs/
  default.yaml              Main region, graph, ERA5, target, model, loss, and training config
  hyperparameter_sweep.yaml Sweep search space and run settings

data/
  raw/                      Raw DEM, ERA5, and ERA5-Land data
  processed/                Static graph, dynamic tensors, targets, visualizations, experiment artifacts

scripts/
  prepare_srtm_nepal.py     Downloads and mosaics SRTM tiles
  visualize_graph_network.py
  visualize_srtm_3d.py

src/
  data/
    build_graph.py          Builds the static terrain graph
    download_era5.py        Downloads ERA5 predictors from CDS
    download_era5land.py    Downloads ERA5-Land target variables from CDS
    download_era5land_precip.py
    preprocess_era5.py      Interpolates ERA5 onto graph nodes
    preprocess_era5land.py  Interpolates ERA5-Land targets onto graph nodes
    build_targets.py        Builds final monthly target tensors
    dataset.py              Runtime dataset for static graph plus dynamic weather

  models/
    layers.py               Reusable GNN layers
    piggn.py                PI-GNN model
    losses.py               Data, lapse-rate, and divergence losses
    mlp_baseline.py         Nodewise MLP baseline

  training/
    train.py                GNN training entry point
    evaluate.py             GNN checkpoint evaluation
    train_baseline.py       Baseline training entry point
    sweep.py                Grid-search driver
    metrics.py              Training and evaluation metrics

  inference/
    predict.py              Inference entry point
    quantile_mapping.py     Post-processing utility

notebooks/
  run_weather_baselines.ipynb
  compare_weather_models.ipynb
```

## Data Design

The project separates static and dynamic data.

Static graph:

```text
data/processed/nepal_graph.pt
```

Contains:

```text
pos          [N, 2]
edge_index   [2, E]
edge_attr    [E, 3]
static_x     [N, 3]
elevation    [N]
slope        [N]
aspect       [N]
metadata
```

Dynamic ERA5 tensors:

```text
data/processed/era5_dynamic_11ch/era5_dynamic_YYYYMM.pt
```

Each monthly file contains:

```text
x_dynamic      [T, N, 6]
time_features  [T, 2]
timestamps     [T]
```

At training time, the dataset combines them into:

```text
x = concat(x_dynamic[t], static_x, time_features[t])
```

Final model input per timestep:

```text
x [N, 11]
```

Feature order:

```text
0  t2m_coarse
1  q850_coarse
2  u10m_coarse
3  v10m_coarse
4  z500_coarse
5  tp_coarse
6  elevation_m
7  slope_rad
8  aspect_rad
9  day_of_year_sin
10 day_of_year_cos
```

Final monthly target tensors:

```text
data/processed/targets/targets_YYYYMM.pt
```

Each monthly file contains:

```text
y             [T, N, 4]
timestamps    [T]
target_names
```

Target order:

```text
0  temperature
1  precipitation
2  u_wind
3  v_wind
```

## Region

The default region is Nepal:

```text
latitude:  26.0 to 31.0
longitude: 80.0 to 89.0
grid step: 0.05 degrees
```

These settings are in:

```text
configs/default.yaml
```

Only rebuild the graph if the region, grid resolution, station list, DEM source, or graph construction settings change.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For ERA5 downloads, configure CDS credentials at:

```text
~/.cdsapirc
```

Expected format:

```yaml
url: https://cds.climate.copernicus.eu/api
key: your-api-key
```

Both `download_era5.py` and `download_era5land.py` use this CDS setup.

## SRTM DEM Preparation

Download and mosaic SRTM tiles for the configured Nepal extent:

```bash
python scripts/prepare_srtm_nepal.py
```

Main output:

```text
data/raw/srtm_nepal.tif
```

The script also saves source tiles and a manifest:

```text
data/raw/srtm_tiles/
data/raw/srtm_nepal_manifest.json
```

## Build Static Graph

Build the static terrain graph:

```bash
python src/data/build_graph.py
```

Output:

```text
data/processed/nepal_graph.pt
```

The graph uses k-nearest neighbors. The default is:

```yaml
graph:
  k_neighbours: 6
  directed: true
```

## Download ERA5

The required ERA5 predictors are:

```text
2m temperature
10m u component of wind
10m v component of wind
total precipitation
specific humidity at 850 hPa
geopotential at 500 hPa
```

Download using the config date range:

```bash
python src/data/download_era5.py
```

Raw monthly files are written to:

```text
data/raw/era5/
```

The download is resumable. Existing files are skipped.

To run it in the background:

```bash
nohup python src/data/download_era5.py > data/raw/era5/download.log 2>&1 &
```

Check progress:

```bash
find data/raw/era5 -name '*.nc' | wc -l
tail -f data/raw/era5/download.log
```

## Download ERA5-Land Targets

The target-side ERA5-Land variables are:

```text
2m temperature
10m u component of wind
10m v component of wind
total precipitation
```

Download the main ERA5-Land variables:

```bash
python src/data/download_era5land.py
```

If precipitation is stored separately in your workflow, download it with:

```bash
python src/data/download_era5land_precip.py
```

Raw monthly files are written to:

```text
data/raw/era5land/
data/raw/era5land_precip/
```

## Preprocess ERA5

Convert raw ERA5 NetCDF files into graph-node dynamic tensors:

```bash
python src/data/preprocess_era5.py
```

Process specific months:

```bash
python src/data/preprocess_era5.py --months 199001 199002
```

Output:

```text
data/processed/era5_dynamic_11ch/
```

The script writes atomically through `.part` files. If a preprocessing run is interrupted, invalid dynamic tensor files are rewritten on the next run.

## Preprocess ERA5-Land Targets

Convert raw ERA5-Land NetCDF files into node-aligned monthly target tensors:

```bash
python src/data/preprocess_era5land.py
```

Output:

```text
data/processed/targets_era5land/
```

Then build the final monthly target packages used by training:

```bash
python src/data/build_targets.py
```

Output:

```text
data/processed/targets/
```

## Dataset Usage

Use `WeatherGraphDataset` to combine the static graph and dynamic monthly tensors:

```python
from src.data.dataset import WeatherGraphDataset

dataset = WeatherGraphDataset(
    "data/processed/nepal_graph.pt",
    "data/processed/era5_dynamic_11ch",
)

sample = dataset[0]

x = sample["x"]
edge_index = sample["edge_index"]
edge_attr = sample["edge_attr"]
timestamp = sample["timestamp"]
```

Each sample represents one timestep:

```text
x          [N, 11]
edge_index [2, E]
edge_attr  [E, 3]
pos        [N, 2]
timestamp  string
```

When target tensors are passed in, the dataset also returns:

```text
y [N, 4]
```

## PI-GNN Model

The PI-GNN is defined in:

```text
src/models/piggn.py
```

The reusable layers are in:

```text
src/models/layers.py
```

The model uses:

- Node encoder
- Edge encoder
- Residual edge-conditioned message passing blocks
- Output head for temperature, precipitation, u-wind, and v-wind

Default model settings:

```yaml
model:
  node_input_channels: 11
  edge_input_channels: 3
  hidden_channels: 128
  edge_hidden_channels: 32
  message_hidden_channels: 128
  output_channels: 4
  num_layers: 4
  dropout: 0.1
```

For one timestep:

```text
input:  [N, 11]
output: [N, 4]
```

Output channels:

```text
0 temperature
1 precipitation
2 u_wind
3 v_wind
```

The default config also enables coarse-field residual paths for temperature, precipitation, and wind.

## Losses

Losses are defined in:

```text
src/models/losses.py
```

Available components:

- Masked MSE
- Masked MAE
- Lapse-rate regularization
- Wind divergence regularization
- Combined `PIGNNLoss`

Current default loss config:

```yaml
loss:
  data_loss: mse
  weights:
    data: 1.0
    lapse_rate: 0.0
    divergence: 0.0
```

## Visualizations

Static graph network:

```bash
python scripts/visualize_graph_network.py
```

High quality vector output:

```bash
python scripts/visualize_graph_network.py --output data/processed/nepal_graph_network.svg
```

SRTM 3D terrain viewer:

```bash
python scripts/visualize_srtm_3d.py
```

## Current Training Plan

The current training flow is:

```text
1. Build static graph once
2. Download ERA5
3. Download ERA5-Land targets
4. Preprocess ERA5 into dynamic tensors
5. Preprocess ERA5-Land targets
6. Build final monthly target tensors
7. Load WeatherGraphDataset
8. Train PI-GNN over timesteps
9. Evaluate baselines and GNN
```

The graph is static. Do not build a new graph for every date. Each timestep uses the same graph topology with different weather features.

Typical commands:

```bash
python src/data/build_graph.py
python src/data/preprocess_era5.py
python src/data/preprocess_era5land.py
python src/data/build_targets.py
python src/training/train.py
python src/training/evaluate.py --checkpoint checkpoints/best.pt
python src/training/train_baseline.py --model interpolation
python src/training/train_baseline.py --model mlp
python src/training/train_baseline.py --model xgboost
python src/training/sweep.py --config configs/default.yaml --sweep-config configs/hyperparameter_sweep.yaml
```

## Notes

- Dynamic ERA5 tensor files are large because they store hourly values for every graph node.
- The default input tensor has 11 node features and the default target tensor has 4 channels.
- Baseline comparison notebooks live in `notebooks/` and expect trained artifacts to exist.
- Config values are stored in `configs/default.yaml`, while sweep settings live in `configs/hyperparameter_sweep.yaml`.
