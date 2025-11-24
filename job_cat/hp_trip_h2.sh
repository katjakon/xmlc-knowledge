#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1 
#SBATCH --time=02:00:00
#SBATCH --licenses=cat 
#SBATCH --mail-type=start,end
#SBATCH --mail-user=katja.konermann@tu-dresden.de
#SBATCH --job-name=hp_trip-h2
#SBATCH --output=hp_trip-h2.out
#SBATCH --error=hp_trip-h2.err

module purge
module load release/24.04  GCCcore/11.3.0
module load Python/3.10.4
export HF_HOME=/data/cat/ws/kako402f-thesis/.cache/
source /data/cat/ws/kako402f-thesis/env/bin/activate
echo "Activated environment"
cd /home/kako402f/projects/cat/kako402f-thesis/xmlc-knowledge

python predict.py --config configs/config_ft_hp_3b_triplets_3k_2h.yaml --result_dir results/ --index search_indices/label_index.pkl  --mapping search_indices/label_mapping.pkl --hard-prompt --seed 42


