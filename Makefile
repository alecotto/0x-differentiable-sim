.PHONY: check static test
static:
	python check_repo.py --static
check:
	python check_repo.py --static
	python -m pytest tests/ -x -q
test:
	python -m pytest tests/ -x -q
