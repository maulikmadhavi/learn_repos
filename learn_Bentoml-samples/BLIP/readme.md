# README

This code is adapted from [BentoBlip](https://github.com/bentoml/BentoBlip).

## Overview

This module provides functionality for [briefly describe what the code does, e.g., "image captioning using BLIP models"].  
It leverages the architecture and design patterns from the original BentoBlip repository, with modifications to suit specific requirements.

## Usage

1. Run service: `bentoml serve .` This starts on port 3000.
2. Run clients: ` python client.py --url http://localhost:3000/generate --requests 200 --concurrency 50`

## Attribution

Original implementation and inspiration from [BentoBlip](https://github.com/bentoml/BentoBlip).