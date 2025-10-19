# Tutorial to reproduce results and figures from the Corgi paper

## Step 1: Downloading the model, data and setting up the virtual environment

1. Clone the repository. `git clone https://github.com/ekinda/corgi-reproduction.git`.
2. Create a new virtual environment called `corgi` at the desired path with `python -m venv corgivenv`.
3. Activate the environment with `source corgivenv/bin/activate` 
4. Install the corgi package from https://github.com/ekinda/corgi using `pip install git+https://github.com/username/repository.git`
5. Install the rest of the required packages with `pip install -r corgi-reproduction/environment.txt`
6. Download the model from `https://zenodo.org/records/17368602` and put it into corgi-reproduction/data
7. Download the data from `` and put it into corgi-reproduction/processed_data

## Step 2a: Reproduction of figures

1. Run the notebooks at figure_notebooks.
