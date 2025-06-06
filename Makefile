.PHONY: install train evaluate clean

install:
	pip install -r requirements.txt

train:
	python src/train.py --config config/config.yaml

evaluate:
	python src/evaluate.py --model-path models/diabetes_model.pkl --test-data data/sample.csv

clean:
	rm -rf __pycache__/ models/ *.pyc
