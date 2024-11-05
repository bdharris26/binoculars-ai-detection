# This script runs experiments for the CC News, CNN, and PubMed datasets with generations from the LLaMA-2-13B model.

# Experiment for the CC News dataset with LLaMA-2-13B model
python run.py \
  --dataset_path ../datasets/core/cc_news/cc_news-llama2_13.jsonl \
  --dataset_name CC-News \
  --human_sample_key text \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B

# Experiment for the CNN dataset with LLaMA-2-13B model
python run.py \
  --dataset_path ../datasets/core/cnn/cnn-llama2_13.jsonl \
  --dataset_name CNN \
  --human_sample_key article \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B

# Experiment for the PubMed dataset with LLaMA-2-13B model
python run.py \
  --dataset_path ../datasets/core/pubmed/pubmed-llama2_13.jsonl \
  --dataset_name PubMed \
  --human_sample_key article \
  --machine_sample_key meta-llama-Llama-2-13b-hf_generated_text_wo_prompt \
  --machine_text_source LLaMA-2-13B
