# PHONY does not check if the file exists or not, it will always run the command
.PHONY: test 
.PHONY: test-coverage
.PHONY: unit-test 
.PHONY: unit-test-coverage-web 
.PHONY: unit-test-coverage 
.PHONY: integration-test 
.PHONY: integration-test-coverage-web
.PHONY: integration-test-coverage
.PHONY: lint
.PHONY: lint-pylint
.PHONY: lint-black
.PHONY: lint-mypy
.PHONY: install-reps
.PHONY: save-deps

test: unit-test integration-test

test-coverage: unit-test-coverage integration-test-coverage

unit-test:
	pytest -v -s ./tests/*_unit_test.py

unit-test-coverage-web:
	coverage run --omit="./tests/*" -m pytest -v -s ./tests/*unit_test.py && coverage html
	explorer.exe `wslpath -w "./htmlcov"`
	# xdg-open ./htmlcov/index.html

unit-test-coverage:
	pytest --cov=deps --cov-report=xml -s ./tests/*unit_test.py

integration-test:
	pytest -v -s ./tests/*_integration_test.py

integration-test-coverage-web:
	coverage run --omit="./tests/*" -m pytest -v -s ./tests/*_integration_test.py && coverage html

integration-test-coverage:
	pytest --cov=deps --cov-report=xml -s ./tests/*_integration_test.py

lint: lint-pylint lint-black lint-mypy

lint-pylint:
	pylint --rcfile=.pylintrc deps/ || true
	pylint --rcfile=.pylintrc cogs/ || true
	pylint --rcfile=tests/.pylintrc tests/ || true

lint-black:
	black **/*.py

lint-mypy:
	mypy deps
	mypy cogs
	mypy tests

install-reps:
	python3 -m pip install -r requirements.txt

save-deps:
	python3 -m pip freeze > requirements.txt
