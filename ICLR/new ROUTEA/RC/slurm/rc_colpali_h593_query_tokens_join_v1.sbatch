#!/usr/bin/env bash
#SBATCH --job-name=colpali_qjoin
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colpali-qjoin-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/colpali-qjoin-%j.err
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
exec '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u programs/cache_colpali_h593_queries_v1.py join
