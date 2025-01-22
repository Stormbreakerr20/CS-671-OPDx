# pip install -q  torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

import os
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# Model and tokenizer names
base_model_name = "llSourcell/medllama2_7b"
refined_model = "opdx"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

data_name = "satyam-03/ddx-conversations-10k"
opdx_data_path = "/neuro/data/opdx/ddx-10k-gpt3/"

if os.path.exists("/neuro/data/opdx/ddx-10k-gpt3/") and not os.listdir("/neuro/data/opdx/ddx-10k-gpt3/"):
    training_data = load_dataset(opdx_data_path, split="train", trust_remote_code=True)
else:
    training_data = load_dataset(data_name, split="train")
    training_data.save_to_disk("/neuro/data/opdx/ddx-10k-gpt3/")

peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=2,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    save_steps=25,
    logging_steps=2,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="tensorboard",
    gradient_checkpointing=True
)

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)
fine_tuning.train()

fine_tuning.model.save_pretrained(refined_model)

fine_tuning.model.save_pretrained('opdx')

# git clone https://github.com/ggerganov/llama.cpp.git
# cd llama.cpp
# pip install -r reqiurement.txt
# python convert-lora-to-ggml.py ../opdx/adapter_config.json

model_name = "llSourcell/medllama2_7b"
adapters_name = 'opdx'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
m = PeftModel.from_pretrained(m, adapters_name)
m = m.to(device)

tok = LlamaTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

inputs = tok("Today was an amazing day because", return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = m.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)

decoded_outputs = tok.batch_decode(outputs, skip_special_tokens=True)
print(decoded_outputs)

torch.save(m.state_dict(), './opdx-full')

prompt = "What are the symptoms of Bronchitis ?"
inputs = tok(prompt, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = m.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)

decoded_outputs = tok.batch_decode(outputs, skip_special_tokens=True)
print(decoded_outputs)


