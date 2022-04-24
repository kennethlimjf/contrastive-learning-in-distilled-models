activate:
	conda activate cl-distilled

setup:
	conda env create -f environment.yml
	conda activate cl-distilled
	pip install -r requirements.txt
	./scripts/download_training_data.sh
	./scripts/download_eval_data.sh
	python -m ipykernel install --user --name=cl-distilled

clean:
	find . | grep -E "\(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	rm -rf `find -type d -name .ipynb_checkpoints`
	rm -rf .mypy_cache/ .pytest_cache/
	rm -rf data/text data/._data*

report:
	cd latex && \
		bash -c "rm -rf main.{aux,bbl,blg,log,pdf}" && \
		pdflatex main && \
		bibtex main  && \
		pdflatex main && \
		pdflatex main
