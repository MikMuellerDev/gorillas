#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=TODO
#SBATCH -A TODO
#SBATCH --qos TODO
#SBATCH --cpus-per-gpu=24
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=100G
#SBATCH --time=4:00:00
#SBATCH --out logs/%j.txt
#SBATCH --container-image=docker://maxscha/research-environment:latest
#SBATCH --container-workdir=/workspace
#SBATCH --container-mounts=PATH_TO_PROJECT:/workspace
#SBATCH --container-writable
#SBATCH --export=ALL


code tunnel  --accept-server-license-terms --name debug --no-sleep