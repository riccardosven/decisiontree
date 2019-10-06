init:
	pip install -r requirements.txt

test:
	python test_decisiontree.py

.PHONY: init test