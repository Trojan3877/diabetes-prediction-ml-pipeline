.PHONY: install train evaluate test clean

install:
	pip install -r requirements.txt

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	pytest tests/ test/ -v

clean:
	rm -rf __pycache__/ models/ *.pyc src/__pycache__/
