# Distilled Face: 

## 1. Setup

1. Create new conda environement: `conda env create -f environment.yml`
2. Activate conda environment: `conda activate cl-distilled`
3. Install pip requirements: `pip install -r requirements.txt`
4. Download training data: `./scripts/download_training_data.sh`
5. Download evaluation data: `./scripts/download_eval_data.sh`
6. Make cl-distilled env available for Jupyter: `python -m ipykernel install --user --name=cl-distilled`

And we're good to go! Distilled Face :)


## 2. Todo

- Discuss entire paper outline
- Finalize training and evaluation methodology
- Add logging training metrics on tensorboard for monitoring
    - loss graph by steps
    - spearman corr eval at every n steps
- Hyperparameter tuning
    - Establish default set of hyperparameters
    - Discuss params to tune and range of values
