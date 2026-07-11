# Physics-Informed GNN Weather Downscaling

This project builds a physics-informed graph neural network for weather downscaling over Nepal. The core idea is to use one static terrain graph and many time-varying atmospheric feature tensors.

The graph represents fixed locations over Nepal. Each node stores terrain information such as elevation, slope, and aspect. Edges connect nearby nodes using k-nearest neighbors and include spatial and terrain relationship attributes. ERA5 weather variables are then interpolated onto those fixed graph nodes for each timestep.

## Project Status

Implemented:

- SRTM DEM download and mosaic for Nepal
- Static terrain graph construction
- ERA5 download script
- ERA5 preprocessing into node-aligned dynamic tensors
- Dataset class that combines static and dynamic features at runtime
- PI-GNN model layers, model definition, and loss functions
- Graph visualization script

Still needed:

- Target data preparation for supervised training
- Full training loop
- Evaluation and plotting utilities
- Experiment configs and checkpoint handling

## Repository Layout

```text
configs/
  default.yaml              Main region, graph, ERA5, model, loss, and training config

data/
  raw/                      Raw DEM and ERA5 data
  processed/                Static graph, dynamic tensors, visualizations

scripts/
  prepare_srtm_nepal.py     Downloads and mosaics SRTM tiles
  visualize_graph_network.py
  visualize_srtm_3d.py

src/
  data/
    build_graph.py          Builds the static terrain graph
    download_era5.py        Downloads ERA5 predictors from CDS
    preprocess_era5.py      Interpolates ERA5 onto graph nodes
    dataset.py              Runtime dataset for static graph plus dynamic weather

  models/
    layers.py               Reusable GNN layers
    piggn.py                PI-GNN model
    losses.py               Data and physics-informed losses

  training/
    train.py                Training entry point placeholder
    evaluate.py             Evaluation entry point placeholder
    metrics.py              Metrics placeholder
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
data/processed/era5_dynamic/era5_dynamic_YYYYMM.pt
```

Each monthly file contains:

```text
x_dynamic      [T, N, 5]
time_features  [T, 2]
timestamps     [T]
```

At training time, the dataset combines them into:

```text
x = concat(x_dynamic[t], static_x, time_features[t])
```

Final model input per timestep:

```text
x [N, 10]
```

Feature order:

```text
0  t2m_coarse
1  q850_coarse
2  u10m_coarse
3  v10m_coarse
4  z500_coarse
5  elevation_m
6  slope_rad
7  aspect_rad
8  day_of_year_sin
9  day_of_year_cos
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
data/processed/era5_dynamic/
```

The script writes atomically through `.part` files. If a preprocessing run is interrupted, invalid dynamic tensor files are rewritten on the next run.

## Dataset Usage

Use `WeatherGraphDataset` to combine the static graph and dynamic monthly tensors:

```python
from src.data.dataset import WeatherGraphDataset

dataset = WeatherGraphDataset(
    "data/processed/nepal_graph.pt",
    "data/processed/era5_dynamic",
)

sample = dataset[0]

x = sample["x"]
edge_index = sample["edge_index"]
edge_attr = sample["edge_attr"]
timestamp = sample["timestamp"]
```

Each sample represents one timestep:

```text
x          [N, 10]
edge_index [2, E]
edge_attr  [E, 3]
pos        [N, 2]
timestamp  string
```

Targets are not prepared yet. Once target tensors exist, the dataset can return:

```text
y [N, target_channels]
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
- Output head for temperature and precipitation

Default model settings:

```yaml
model:
  node_input_channels: 10
  edge_input_channels: 3
  hidden_channels: 128
  edge_hidden_channels: 32
  message_hidden_channels: 128
  output_channels: 2
  num_layers: 4
  dropout: 0.1
```

For one timestep:

```text
input:  [N, 10]
output: [N, 2]
```

Output channels:

```text
0 temperature
1 precipitation
```

Precipitation is passed through a non-negative output transform.

## Losses

Losses are defined in:

```text
src/models/losses.py
```

Available components:

- Masked MSE
- Masked MAE
- Lapse-rate regularization
- Precipitation non-negativity penalty
- Edge smoothness penalty
- Combined `PIGNNLoss`

The default config keeps physics regularizers off until target data and baseline behavior are validated:

```yaml
loss:
  weights:
    data: 1.0
    lapse_rate: 0.0
    precipitation_nonnegative: 0.0
    smoothness: 0.0
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

The intended training flow is:

```text
1. Build static graph once
2. Download ERA5
3. Preprocess ERA5 into dynamic tensors
4. Prepare target tensors
5. Load WeatherGraphDataset
6. Train PI-GNN over timesteps
7. Evaluate spatial and temporal skill
```

The graph is static. Do not build a new graph for every date. Each timestep uses the same graph topology with different weather features.

## Notes

- Dynamic ERA5 tensor files are large because they store hourly values for every graph node.
- The model currently predicts two target channels, temperature and precipitation.
- Supervised training cannot start until target data has been prepared.
- Config values are stored in `configs/default.yaml`. Architecture implementation remains in code, while experiment settings should be stored in config.
