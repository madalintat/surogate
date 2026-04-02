#!/bin/bash

docker build -t ghcr.io/invergent-ai/sky-llama-cpp:full-cuda12 -f Dockerfile .
docker push ghcr.io/invergent-ai/sky-llama-cpp:full-cuda12
