# Everyday commands. `make help` lists them.
# Everything runs inside the containers, so the only thing you need installed is Docker.

COMPOSE := docker compose
RUN := $(COMPOSE) run --rm api
N ?= 2000
RPS ?= 100
OFFSET ?= 0

.PHONY: help setup up down restart logs simulate profiles replay export reset test redis-cli psql shell

help: ## list commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-10s %s\n", $$1, $$2}'

setup: up simulate profiles ## first run: start services, generate data, load batch features

up: ## build and start the api, redis and postgres
	$(COMPOSE) up -d --build

down: ## stop everything (the decision log in postgres is kept)
	$(COMPOSE) down

restart: ## recreate the api container, picking up .env changes such as a new MODEL_PATH
	$(COMPOSE) up -d --force-recreate api

logs: ## follow the api logs
	$(COMPOSE) logs -f api

simulate: ## generate simulated transactions into data/
	$(RUN) python -m simulator.generate

profiles: ## the nightly batch job: load card profiles and the blocklist into redis
	$(RUN) python -m jobs.build_profiles

replay: ## send live transactions to the api, e.g. make replay N=5000 RPS=200 OFFSET=0
	$(RUN) python -m scripts.replay --url http://api:8000 --offset $(OFFSET) --limit $(N) --rps $(RPS) --truth data/ground_truth.parquet

export: ## turn the decision log into training data (data/training/)
	$(RUN) python -m jobs.export_training_data

reset: ## empty redis and the decision log, then reload batch features
	$(RUN) python -m scripts.reset
	$(RUN) python -m jobs.build_profiles

test: ## run the test suite
	$(RUN) pytest -q

redis-cli: ## open a redis shell
	$(COMPOSE) exec redis redis-cli

psql: ## open a postgres shell
	$(COMPOSE) exec postgres sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

shell: ## open a bash shell in a container with the project code
	$(RUN) bash
