# This Makefile is just for building the Docker image for testing impsy
# Normal users don't need to worry about this
DOCKERHUB_USERNAME = charlepm
CONTAINER_NAME = impsy
CONTAINER_VERSION = 0.5.3

build: Dockerfile
	docker build --tag $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION) --platform linux/amd64,linux/arm64 -f Dockerfile ..

publish: build
	docker push $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION)

run: build
	docker run -it $(DOCKERHUB_USERNAME)/$(CONTAINER_NAME):$(CONTAINER_VERSION) poetry run ./start_impsy.py run
