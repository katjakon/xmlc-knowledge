#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1 
#SBATCH --time=48:00:00
#SBATCH --mail-type=start,end
#SBATCH --mail-user=katja.konermann@tu-dresden.de
#SBATCH --job-name=ft-ret
#SBATCH --output=ft-ret.out
#SBATCH --error=ft-ret.err

module load release/23.10 GCCcore/11.3.0
export HF_HOME=/home/kako402f/projects/cat/kako402f-thesis-cat/.cache/
source /data/horse/ws/kako402f-thesis/ki-env/bin/activate
echo "Activated environment"
cd /home/kako402f/projects/cat/kako402f-thesis-cat/xmlc-knowledge

python finetune_fs_retriever.py --config config_retriever/few_shot.yaml 


