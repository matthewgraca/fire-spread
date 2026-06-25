# Description
CSULA's AI and Data Science Laboratory development on machine learning for fire spread prediction.

# Install
Dependencies are managed by conda. Run `conda env create -f x_environment.yml`
- `cpu_environment` is my laptop environment. Obviously, not doing any training.
- `gpu_enviornment` will contain by desktop environment which I use to train.

Configure environment while in the project source directory: `python -m pip install -e .`
- This means no more editing `sys.path`, files will be properly exposed from the root directory since this will be a proper project.
