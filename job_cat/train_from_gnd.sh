#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1 
#SBATCH --time=40:00:00
#SBATCH --mail-type=start,end
#SBATCH --mail-user=katja.konermann@tu-dresden.de
#SBATCH --job-name=train-gnd
#SBATCH --output=train-from-gnd.out
#SBATCH --error=train-from-gnd.err
#SBATCH --licenses=cat

module purge
module load release/24.04  GCCcore/11.3.0
module load Python/3.10.4
export HF_HOME=/data/cat/ws/kako402f-thesis/.cache/
source /data/cat/ws/kako402f-thesis/env/bin/activate
echo "Activated environment"
cd /home/kako402f/projects/cat/kako402f-thesis/xmlc-knowledge

python main.py --config configs/config_pt_from_gnd_8b.yaml --load_from_pretrained pt_models/prompt-tuning-gnd-neighbor2label-nohidden-8B/best_model/model.safetensors
