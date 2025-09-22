# marl-control-auv

Simulation and implementation of Multi-Agent Reinforcement Learning (MARL) algorithms for coordinating Autonomous Underwater Vehicles (AUVs) in Internet of Underwater Things (IoUT) scenarios with an emphasis on energy efficiency.  
This repository contains training and evaluation scripts for a multi-AUV coordination system using MARL algorithms, specifically **MADDPG** and **MAPPO**.

---

## File Structure

| File / Folder          | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `train_multi_agent.py` | MADDPG implementation (initial version), saves to `saved_models/maddpg/`   |
| `train_mappo.py`       | MAPPO implementation with improved simulation                             |
| `eval.py`              | General evaluation script, use `--algo` and `--model_path` arguments        |
| `plot_eval.py`         | Script for visualizing evaluation metrics                                  |
| `saved_models/`        | Automatically stores trained models (subfolders: `maddpg/`, `mappo/`)     |

---

## How to Train

### 1. MADDPG
```bash
python train_multi_agent.py
### 2. MAPPO
```bash
python train_mappo.py

## How to Evaluate

### 1. Evaluate MADDPG
```bash
python eval.py --algo maddpg --model_path saved_models/maddpg --episodes 100 --render
### 2. Evaluate MAPPO
```bash
python eval.py --algo mappo --model_path saved_models/mappo --episodes 100 --render

### Plot Results
```bash
python plot_eval.py

### Model Output Directory
saved_models/
├── maddpg/
└── mappo/

### Citation
If this repository helps you in your academic research, you are encouraged to cite our paper.
Here is an example BibTeX entry:
```python
@ARTICLE{Wibisono2025Survey,
  author={Wibisono, Arif and Song, Hyoung-Kyu and Lee, Byung Moo},
  journal={IEEE Access},  
  title={A Survey of Multi-Agent Reinforcement Learning for Cooperative Control in Multi-AUV Systems},  
  year={2025},  
  volume={13},  
  pages={161505-161528},  
  issn={2169-3536},  
  doi={10.1109/ACCESS.2025.3609457}
}
```


