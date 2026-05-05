# Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
#           certora-cloud CLI: uv tool install certora-cloud

.PHONY: login
login:
	certora-cloud codeartifacts login

.PHONY: install
install: login
	uv sync --group ml

.PHONY: update-deps
update-deps: login
	uv lock --upgrade
	uv sync --group ml

.PHONY: build
build:
	uv build
