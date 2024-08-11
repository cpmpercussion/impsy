# This Makefile is just for building the Docker image for testing impsy
# Normal users don't need to worry about this
DOCKERHUB_USERNAME = charlepm
CONTAINER_NAME = impsy
CONTAINER_VERSION = latest

CONTAINER = $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION)

build: Dockerfile
	docker build --tag $(CONTAINER) --platform linux/amd64,linux/arm64 -f Dockerfile ..

publish: build
	docker push $(CONTAINER)

run: build
	docker run -it --rm $(CONTAINER) poetry run ./start_impsy.py run
