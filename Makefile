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

run-db:
	docker run -d \
	--name trading-clickhouse \
	-p 8123:8123 \
	-p 9000:9000 \
	-e CLICKHOUSE_DB=trading_bot \
	-e CLICKHOUSE_USER=trading_user \
	-e CLICKHOUSE_PASSWORD=trading_password_123 \
	-e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
	--ulimit nofile=262144:262144 \
	-v trading_clickhouse_data:/var/lib/clickhouse \
	clickhouse/clickhouse-server:latest