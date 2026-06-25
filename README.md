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
- `scripts/gofer` contains driver code that runs certain aspects of the gofer pipeline. For example, `demo_ortho.py` demonstrates the orthorectification pipeline, producing output for you to evaluate.

`viz` contains various utilities for visualizing certain aspects of our pipeline. For instance:
- `viz/gofer/ortho.py` contains the utilities for visualizing the output of the orthorectification pipeline.
