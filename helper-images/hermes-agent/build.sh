#!/bin/bash

docker build -t ghcr.io/invergent-ai/hermes-agent:latest -f Dockerfile .
docker push ghcr.io/invergent-ai/hermes-agent:latest
