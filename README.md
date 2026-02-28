# Reproducing figures from the Corgi paper

This repository includes ready-to-run reproduction notebooks under `reproduce_notebooks/`.

## Quick start

1. Clone this repository and enter it:

	```bash
	git clone https://github.com/ekinda/corgi-reproduction.git
	cd corgi-reproduction
	```

2. Create and activate a Python environment:

	```bash
	python -m venv .venv
	source .venv/bin/activate
	```

3. Install notebook dependencies:

	```bash
	pip install -r requirements-reproduction.txt
	```

4. Download `reproduction_data.tar.gz` from Zenodo and extract it in the repository root.

	- Zenodo URL (to be updated):
	- After extraction, this path must exist:

	```
	corgi-reproduction/reproduction_data/
	```

5. Run the notebooks:

	```bash
	cd reproduce_notebooks
	jupyter lab
	```

	Then run all cells for:

	- `fig2.ipynb`
	- `fig3.ipynb`
	- `fig4.ipynb`
	- `fig5.ipynb`
	- `fig6.ipynb`
	- `fig7.ipynb`

## Notes

- The reproduction notebooks are configured to read inputs from `../reproduction_data/...`.
