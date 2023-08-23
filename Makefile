PROJECT_NAME := next

BUILD_TAG := latest

export PROJECT_NAME
export BUILD_TAG

help: ## display all options of the Makefile
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## build the base docker with the DeepAR environments
	docker build \
	  -f docker/dockerfiles/Dockerfile \
	  --network host \
	  -t $(PROJECT_NAME):$(BUILD_TAG) .

run: stop ## run the deepar docker
	docker/scripts/run-docker.sh

enter: ## enter a terminal session within a running deepar docker
	docker/scripts/enter-docker.sh

stop: ## stop the running deepar docker
	docker/scripts/stop-docker.sh
