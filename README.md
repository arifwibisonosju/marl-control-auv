# marl-control-auv
This repository provides training and evaluation scripts for a multi-AUV system using the MARL algorithms MADDPG and MAPPO.

python train_multi_agent.py
python train_mappo.py

All three will save to saved_models/ .. maddpg/mappo/moddpg

# Evaluate MADDPG
python eval.py --algo maddpg --model_path saved_models/maddpg --episodes 100 --render

# Evaluate MAPPO
python eval.py --algo mappo --model_path saved_models/mappo --episodes 100 --render

# Plot
python plot_eval.py
