#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1 
#SBATCH --time=30:00:00
#SBATCH --mail-type=start,end
#SBATCH --mail-user=katja.konermann@tu-dresden.de
#SBATCH --job-name=train_gnd
#SBATCH --output=train_gnd.out
#SBATCH --error=train_gnd.err
#SBATCH --licenses=cat

module purge
module load release/24.04  GCCcore/11.3.0
module load Python/3.10.4
export HF_HOME=/data/cat/ws/kako402f-thesis/.cache/
source /data/cat/ws/kako402f-thesis/env/bin/activate
echo "Activated environment"
cd /home/kako402f/projects/cat/kako402f-thesis/xmlc-knowledge

python main_gnd.py --config configs/config_pt_gnd_8b.yaml 
