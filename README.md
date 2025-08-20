<div align="center">

# RL-Guider: Leveraging Historical Decisions and Feedback for Drug Editing with Large Language Models 

<p align="center">

  <a href="https://aclanthology.org/2025.findings-acl.680/">
    <img src="https://img.shields.io/badge/ACL 2025-Findings-b31b1b.svg" alt="arXiv">
  </a>
</p>
</div>


<div align="center">
<img src="main_fig.jpg" alt="framework" width="800">

**RL_Guider Framework**
</div>

## 🛠 Installation


<details>
<summary>
Please follow the installation guideline and prepare the environment
</summary>

```bash
conda create -n rl-guider python=3.9
conda activate rl-guider
pip install -r requirements.txt
```
</details>


## 💡 Module Preparation
***Download neccesary module***

<details>
<summary>
Download Pre-trained Protein Model
</summary>

```bash
cd ./rl-guider
python download.py
```
</details>

<details>
<summary>
Prepare mhcflurry module for peptide analysis
</summary>
    
```bash
mhcflurry-downloads fetch models_class1_presentation
mv mhcflurry-downloads path models_class1_presentation /Data/peptide/models_class1_presentation
```
</details>


## 🎯 Start Drug Editing!
<details>
<summary>
Gather Buffer
</summary>
  
```bash
# Proceed under 'rl_guider' folder
python gather_buffer_smiles.py --num_of_episode=2
```
</details>

<details>
<summary>
Process Buffer
</summary>
  
```bash
python process_buffer_smiles.py --replay_buffer_name='general_replay_buffer_mol_epi_2'
```
</details>

<details>
<summary>
Train RL Guider
</summary>
hi

```bash
python train_rl_smiles.py --task_id=101 --replay_buffer_name='general_replay_buffer_mol_epi_2' --constraint='strict' --reward_type='add' --a=1 --b=1 --c=0 --tau=0.01
```
</details>

<details>
<summary>
Run RL Guider 
</summary>

```bash
python train_rl_smiles.py --task_id=101 --replay_buffer_name='general_replay_buffer_mol_epi_2' --constraint='strict' --reward_type='add' --a=1 --b=1 --c=0 --tau=0.01
```
</details>



## Citation
```bibtex
@inproceedings{liu-etal-2025-rl,
    title = "{RL}-Guider: Leveraging Historical Decisions and Feedback for Drug Editing with Large Language Models",
    author = "Liu, Xufeng  and
      Ding, Yixuan  and
      Qu, Jingxiang  and
      Zhang, Yichi  and
      Gao, Wenhan  and
      Liu, Yi",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    url = "https://aclanthology.org/2025.findings-acl.680/",
    doi = "10.18653/v1/2025.findings-acl.680",
    pages = "13121--13138",
    ISBN = "979-8-89176-256-5"
}
```

Thanks for your interest in our work!
