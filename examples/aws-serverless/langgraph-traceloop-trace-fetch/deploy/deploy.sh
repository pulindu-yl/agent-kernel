#!/bin/bash
set -eo pipefail # exit if any command in this script fails

# Deploys as a container image (Terraform builds and pushes it from ../dist), so Docker must be running.
#
# Pass `local` as the first argument to install agentkernel from ../../../ak-py/dist instead of PyPI. The local
# wheel is force-reinstalled with --no-index: it carries the same version as a published release, and the example
# needs the local code (Trace.fetch_spans).

create_deployment_package() {
	pushd ../
	rm -rf dist
	mkdir -p dist/data
	uv export --no-dev --no-hashes >requirements.txt
	uv pip install --compile-bytecode -r requirements.txt --target=dist/data
	if [[ ${LOCAL_BUILD-} == "local" ]]; then
		uv pip install --compile-bytecode --force-reinstall --no-deps --no-index --target=dist/data --find-links ../../../ak-py/dist agentkernel
	fi
	cp -r lambda.py fetch_api.py config.yaml dist/data
	rm -f requirements.txt
	popd || exit 1
	cp Dockerfile ../dist/
}

LOCAL_BUILD=${1-}

create_deployment_package

terraform init
terraform apply
