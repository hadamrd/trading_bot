.PHONY: lint clean-imports sort-imports fix-all

lint:
	poetry run ruff check src/

clean-imports:
	poetry run autoflake --in-place --recursive --remove-unused-variables --remove-all-unused-imports src/

sort-imports:
	poetry run isort src/

fix-all: clean-imports sort-imports
	poetry run ruff check src/ --fix
	poetry run ruff check src/