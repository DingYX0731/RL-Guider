from huggingface_hub import snapshot_download

snapshot_download(repo_id="chao1224/ChatDrug_data", repo_type="dataset", local_dir="Data", local_dir_use_symlinks=False, ignore_patterns=["README.md"])

from huggingface_hub import hf_hub_download

hf_hub_download(
  repo_id="chao1224/ProteinCLAP_pretrain_EBM_NCE_downstream_property_prediction",
  repo_type="model",
  filename="pytorch_model_ss3.bin",
  cache_dir="Data/protein")



