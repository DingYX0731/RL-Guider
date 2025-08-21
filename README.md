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

# 

# 📰 News
- 2025.04 Our paper has been accepted by ACL 2025 Findings!

# ✒️ Abstract
Recent success of large language models (LLMs) in diverse domains showcases their potential to revolutionize scientific fields, including drug editing. Traditional drug editing relies on iterative conversations with domain experts, refining the drug until the desired property is achieved. This interactive and iterative process mirrors the strengths of LLMs, making them well-suited for drug editing. *In existing works, LLMs edit each molecule independently without leveraging knowledge from past edits.* However, human experts develop intuition about effective modifications over time through historical experience; accumulating past knowledge is pivotal for human experts, and so it is for LLMs. *In this work, we propose RL-Guider — a reinforcement-learning agent to provide suggestions to LLMs; it uses the rich information provided from evaluating editing results made by the LLM based on the recommendations to improve itself over time.* RL-Guider is the first work that leverages both the comprehensive “world-level” knowledge of LLMs and the knowledge accumulated from historical feedback. As a result, RL-Guider mitigates several shortcomings of existing approaches and demonstrates superior performance. 

# 🚀 Running Experiments

## Installation
It is recommended to use Conda to manage the environment.
```bash
conda create -n rl-guider python=3.9
conda activate rl-guider
pip install -r requirements.txt
```

## API Key
Configure your API key in `src/llm/deepseek_interface.py`:
```python
API_KEY = 'your-api-key'
```

## Running
The project execution is divided into three main steps:

### 1. Train the GNN Encoder
This step trains the LightGCN model to generate embeddings for users and spatiotemporal contexts. These embeddings are crucial for retrieving relevant historical records.
```bash
python Encoder.py
```
This script will train the model and save the embeddings and mappings in the `./model_output/` directory.

### 2. Predict Living Needs with LLM
This step uses the trained GNN encoder to retrieve relevant records and then leverages an LLM to predict living needs in an open-set manner.
```bash
python PIGEON.py
```
The script will perform the following actions:
1.  Load the pre-trained embeddings.
2.  For each entry in the test set, retrieve relevant personal and similar users' historical records.
3.  Use a large language model (GPT-4o-mini by default) to predict the living need based on the retrieved records.
4.  Refine the prediction using Maslow's hierarchy of needs.
5.  Save the results to a CSV file (e.g., `llm_results_YYYYMMDD_HHMMSS.csv`).
6.  Use a fine-tuned sentence transformer model to recall relevant services based on the predicted need.
7.  Evaluate the recall performance using NDCG and Recall@k metrics.

### 3. Fine-tune the Recall Model
To adapt the recall model to flexible need descriptions, you first need to generate refined predictions.

**3.1. Generate Refined Predictions for Fine-tuning Data**
Assuming you have a file `llm_results_finetune.csv` with `order_intention` and `predicted_intention` columns, run `Query.py` to add a `refined_prediction` column.
```bash
python Query.py
```
This will generate `llm_results_finetune_refined.csv`.

**3.2. Fine-tune the Sentence Transformer Model**
This step fine-tunes a text embedding model (e.g., `BAAI/bge-base-zh-v1.5`) to better map the flexible living need descriptions to specific life services.
```bash
python fine_tuning.py
```
The script will:
1.  Load the refined prediction data.
2.  Construct triplet training examples (anchor, positive, negative).
3.  Fine-tune the sentence transformer model.
4.  Save the best model to the `output_model/best_model_triplet_loss_llm_refined` directory, which can then be used in `PIGEON.py` for evaluation.


# 🌟 Citation

If you find this work helpful, please cite our paper:

```latex
@article{lan2025open,
  title={Open-Set Living Need Prediction with Large Language Models},
  author={Lan, Xiaochong and Feng, Jie and Sun, Yizhou and Gao, Chen and Lei, Jiahuan and Shi, Xinlei and Luo, Hengliang and Li, Yong},
  journal={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
```

# 📩 Contact

If you have any questions or want to use the code, feel free to contact:
Jie Feng (fengjie@tsinghua.edu.cn)





## 🛠 Installation


<details>
<summary>
Please follow the installation guideline and prepare the environment
</summary>

```bash

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
python run_planner_tree.py --conversational_LLM='deepseek' --depth=3 --num_generate=1 --num_keep=1 --num_of_mol=200 --task_id=101 --planner='baseline' --constraint='strict' --conversation_type='single'
```
</details>



## Citation
```bibtex
@inproceedings{liu-etal-2025-rl,
    title = "{RL}-Guider: Leveraging Historical Decisions and Feedback for Drug Editing with Large Language Models",
    author = "Liu, Xufeng  and Ding, Yixuan  and Qu, Jingxiang  and Zhang, Yichi  and Gao, Wenhan  and Liu, Yi",
    booktitle  = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
}
```

Thanks for your interest in our work!
