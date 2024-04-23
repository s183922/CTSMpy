.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-pyc clean-test ## remove all build, test, coverage and Python artifacts


clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .ruff_cache/

lint/flake8: ## check style with flake8
	flake8 ctsmpy tests


lint: lint/flake8 ## check style

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source ctsmpy -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

test_lint:
	mypy --strict  src/
	ruff check .


setup_env: ## setup the environment for development
	pip install -r requirements_dev.txt
	pip
	pip install -e .

