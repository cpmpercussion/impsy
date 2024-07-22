# This Makefile is just for building the Docker image for testing impsy
# Normal users don't need to worry about this
DOCKERHUB_USERNAME = charlepm
CONTAINER_NAME = impsy
CONTAINER_VERSION = 0.5.1

build: Dockerfile
	docker build --tag $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION) -f Dockerfile ..

push: build
	docker push $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION)

run: build
	docker run -it $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION) bash
