#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1 
#SBATCH --time=03:30:00
#SBATCH --mail-type=start,end
#SBATCH --mail-user=katja.konermann@tu-dresden.de
#SBATCH --job-name=con-hp
#SBATCH --output=con-hp.out
#SBATCH --error=con-hp.err

module load release/23.10 GCCcore/11.3.0
export HF_HOME=/data/cat/ws/kako402f-thesis-cat/.cache
source /data/horse/ws/kako402f-thesis/ki-env/bin/activate
echo "Activated environment"
cd /home/kako402f/projects/cat/kako402f-thesis-cat/xmlc-knowledge

python predict.py --config configs/config_ft_hp_8b_context_label_5.yaml --result_dir results/ --index search_indices/finetuned-label_index.pkl  --mapping search_indices/finetuned-label_mapping.pkl --hard-prompt


