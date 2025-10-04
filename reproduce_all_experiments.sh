#!/bin/bash

# Main entry point for reproducing MISL paper experiments
# Usage: ./reproduce_all_experiments.sh [arithmetic|gsm8k|all]

set -e

cd scripts
./reproduce_experiments.sh "$@"