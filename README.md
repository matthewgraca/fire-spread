# Description
CSULA's AI and Data Science Laboratory development on machine learning for fire spread prediction.

# Install
Dependencies are managed by conda. Run `conda env create -f x_environment.yml`
- `cpu_environment` is my laptop environment. Obviously, not doing any training.
- `gpu_enviornment` will contain by desktop environment which I use to train.

Configure environment while in the project source directory: `python -m pip install -e .`
- This means no more editing `sys.path`, files will be properly exposed from the root directory since this will be a proper project.

# Rundown of repo
`src/gofer` contains our implementation of the GOFER algorithm.

`scripts` contains various useful scripts that don't exactly fit either `src` or `tests`, that run certain aspects of our pipeline. For instance:
- `scripts/gofer` contains driver code that runs the GOFER pipeline.
- `scripts/gofer/demo` contains scripts the run individual aspects of the pipeline, used for visual verification. For example, `demo_ortho.py` demonstrates the orthorectification pipeline, producing output for you to evaluate.

`viz` contains various utilities for visualizing certain aspects of our pipeline. For instance:
- `viz/gofer/ortho.py` contains the utilities for visualizing the output of the orthorectification pipeline.

# GOFER Usage
The GOFER script is a complete end-to-end script that handles ingestion down the generation of visualizations. To use this script, you'll need three things:
1. Manifest
    - This is defined in `manifests/fires.csv` where you define the fires you want. These fires must exist in the CalFire dataset, found [here](https://data.ca.gov/dataset/california-fire-perimeters-all).
    - It is recommended you create your own local manifest (`cp manifest/fires.csv manifest/local_fires.csv`), then pick and choose the fires you want to ingest.
2. Configuration
    - This is defined in `configs/gofer.yaml` where you define various knobs, input file paths, etc. The readme found [here](https://github.com/matthewgraca/fire-spread/blob/main/configs/README.md) will teach you what variable does what.
    - It is recommended you create your own local config (`cp configs/gofer.yaml configs/local_gofer.yaml`), then tune your parameters.
3. Execution
    - To execute the script, call `python scripts/gofer/run --config configs/<your_config>.yaml`
    - It is generally recommended you split the ingest and the GOFER pipeline, as each step takes a significant amount of time. You can do so by using the arguments:
        - `python scripts/gofer/run.py --config configs/<your_config>.yaml --only-ingest`
        - `python scripts/gofer/run.py --config configs/<your_config>.yaml --skip-ingest`

To get a clearer view of the pipeline, check out the readme [here](https://github.com/matthewgraca/fire-spread/blob/main/scripts/gofer/README.md).

For a quick overview of the GOFER algorithm, check out the readme [here](https://github.com/matthewgraca/fire-spread/blob/main/src/gofer/README.md).

For a full examination of the algorithm, refer to the original GOFER paper [here](https://essd.copernicus.org/articles/16/1395/2024/).
