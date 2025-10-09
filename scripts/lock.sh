#!/bin/bash -e

conda-lock --micromamba -p linux-64 --with-cuda=12.8 --lockfile conda-lock.yml
conda-lock --micromamba -p linux-aarch64 --with-cuda=12.8 --lockfile arm64.conda-lock.yml
