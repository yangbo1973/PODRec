# **Partial-Order-Aware Diffusion Model for Multi-Behavior Recommendation**



### Overview

This code appendix contains the implementation of **PODRec (Partial-Order-Aware Diffusion Model for Multi-Behavior Recommendation)**, a novel framework that bridges diffusion models with multi-behavior recommendation tasks. PODRec introduces two key innovations:

1. **Partial-Order-Guided Grouping of Behavior Union Graphs (PGBUG)**: A structured behavior fusion framework that preserves semantic hierarchies among different user behaviors (e.g., click → cart → buy).
2. **Behavior-Specific Diffusion Paths**: A tailored denoising process that treats target behaviors as clean signals while considering auxiliary behaviors as structured noise, enabling precise multi-behavior recommendation.


### Datasets

The Taobao and Tmall datasets can be downloaded from the BCIPM model repository:

https://github.com/MingshiYan/BIPN

The `data.zip` file in this repository contains the required data. After downloading, please run `data_process.py` in the corresponding data directory for preprocessing, following the instructions of BIPN.


### Implementation

##### Set up a new environment with the following requirements.

```
python==3.9.13
loguru==0.7.3
numpy==1.22.4
pandas==2.2.3
scikit-learn==1.5.2
scipy==1.6.0
setuptools==75.1.0
tensorboard==2.18.0
torch==2.4.1+cu121
tqdm==4.66.5
```

##### Train the PODRec with specific hyperparameters.

- For the Taobao dataset:

```
python main.py --model_name PODRec --model_path ./checkpoint --data_path ./data --log_path ./log --gpu 0 --seed 2023 --lr 0.001 --decay 0.0 --epochs 100 --data_name taobao --early_stop 25 --batch_size 2048 --test_batch_size 2048 --training_interval 5 --if_multi_tasks --aux_weight_loss 1.0 --aux_weight_score 0.6 --no_impact no_action --time_type cat --dnn_dims "[8000]" --timestep_emb_size 10 --diffusion_steps 40 --noise_schedule linear-var --noise_scale 0.001 --noise_min 0.0005 --noise_max 0.005 --sampling_steps 0 --reweight True
```

- For the Tmall dataset:

```
python main.py --model_name PODRec --model_path ./checkpoint --data_path ./data --log_path ./log --gpu 0 --seed 2021 --lr 0.001 --decay 0.0 --epochs 100 --data_name tmall --early_stop 25 --batch_size 1024 --test_batch_size 2048 --training_interval 5 --if_multi_tasks --aux_weight_loss 0.6 --aux_weight_score 0.2 --no_impact no_action --time_type cat --dnn_dims "[8000]" --timestep_emb_size 10 --diffusion_steps 40 --noise_schedule linear-var --noise_scale 0.001 --noise_min 0.0005 --noise_max 0.005 --sampling_steps 0 --reweight True
```

##### Check the results.

- Logging files: check `./log` and select the folder named by timestamp.
- Checkpoint files: check `./checkpoint` and select the folder named by its timestamp, and access `model.pth` and `all_embeddings.pth`.