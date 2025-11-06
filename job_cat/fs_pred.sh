#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1 
#SBATCH --time=03:30:00
#SBATCH --mail-type=start,end
#SBATCH --mail-user=katja.konermann@tu-dresden.de
#SBATCH --job-name=fs_tit
#SBATCH --output=fs_tit.out
#SBATCH --error=fs_tit.err

module purge
module load release/24.04  GCCcore/11.3.0
module load Python/3.10.4
export HF_HOME=/data/cat/ws/kako402f-thesis-cat/.cache/
source /data/cat/ws/kako402f-thesis-cat/xmlc-knowledge/env/bin/activate
echo "Activated environment"
cd /home/kako402f/projects/cat/kako402f-thesis-cat/xmlc-knowledge

python fs_predict.py --config configs/config_fs_title_5k.yaml --result_dir results/ --index search_indices/label_index.pkl  --mapping search_indices/label_mapping.pkl --example-type title --seed 42 
