# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/NAS/luyq/huggingface/hub")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/NAS/luyq/huggingface/hub")