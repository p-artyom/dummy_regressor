test:
	python test_extension.py

style:
	black -S -l 79 .
	isort .
	flake8 .
