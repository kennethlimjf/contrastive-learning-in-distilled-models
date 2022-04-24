# Distilled Face: 

## 1. Setup

1. Create new conda environement: `conda env create -f environment.yml`
2. Activate conda environment: `conda activate cl-distilled`
3. Install pip requirements: `pip install -r requirements.txt`
4. Download training data: `./scripts/download_training_data.sh`
5. Download evaluation data: `./scripts/download_eval_data.sh`
6. Make cl-distilled env available for Jupyter: `python -m ipykernel install --user --name=cl-distilled`

And we're good to go! Distilled Face :)


## 2. Notebooks and Source Code

1. Source code for modules required by DistilFACE can be found in:

- `src/distilface/modules/pooler.py`
- `src/distilface/modules/similarity.py`

2. Main DistilFACE model implementation can be found in the notebooks in folder:

- Section 2 of `notebooks/training/*.ipynb`

3. Hyperparameter tuning results can be found in all the notebooks in:

- Section 4 of `notebooks/training/*.ipynb`


## 3. LaTEX Setup

1. Install required binaries:

```bash
$ apt-get update && apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-bibtex-extra
```

2. Generate LaTeX Report:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Referenced from [Bibliography_management_with_bibtex](https://www.overleaf.com/learn/latex/Bibliography_management_with_bibtex)
