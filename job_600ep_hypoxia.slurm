#!/bin/bash
#SBATCH --job-name=bios_hypoxia_600
#SBATCH --output=logs/bios_hypoxia_600_%j.out
#SBATCH --error=logs/bios_hypoxia_600_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "=========================================="
echo "BIOS: 3+Hypoxia, 600 epochs, Linear"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC

mkdir -p logs
mkdir -p results
mkdir -p save/model_combo_linear_3plus_hypoxia_600ep

python3 train_bioanchor.py \
    --cancer BRCA \
    --bio_anchor_file bio_anchors_BRCA_combo_3plus_hypoxia_score.csv \
    --bio_dim 4 \
    --head_type linear \
    --lambda_bio 0.1 \
    --epochs 600 \
    --save_dir save/model_combo_linear_3plus_hypoxia_600ep \
    --batch_size 32 \
    --lr 3e-4

echo ""
echo "Training complete. Running evaluation..."
echo ""

python3 evaluate_clustering.py \
    --cancer BRCA \
    --model_path save/model_combo_linear_3plus_hypoxia_600ep/checkpoint_600.tar \
    --bio_dim 4 \
    --head_type linear \
    --result_prefix results/BRCA_hypoxia_600ep

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
