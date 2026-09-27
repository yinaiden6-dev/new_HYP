#!/usr/bin/env bash
#SBATCH --job-name=rebut_QR2_join
#SBATCH --partition=cpuonly
#SBATCH --account=hk-project-p0025545
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut-QR2-join-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new\ ROUTEA/RC/logs/rebut-QR2-join-%j.err
#SBATCH --export=NIL
set -euo pipefail
unset LD_LIBRARY_PATH
export PATH=/usr/local/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
cd '/hkfs/work/workspace/scratch/ap7811-benchmark'
exec '/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python' -u 'ICLR/new ROUTEA/RC/programs/join_rebut_qr_qrr_frozenbase_v2.py' --protocol 'ICLR/new ROUTEA/RC/results/rc_rebut_qr_qrr_frozenbase_v2/protocol.json'
