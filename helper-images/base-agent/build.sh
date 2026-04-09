#!/bin/bash

docker build -t ghcr.io/invergent-ai/base-agent-image:latest -f Dockerfile .
docker push ghcr.io/invergent-ai/base-agent-image:latest
