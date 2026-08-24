"""
Khatta-ka AI: Khattak Dialect Pashto LLM
Fine-tuned using Unsloth and Hugging Face.
"""

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "junaid008/qehwa-pashto-llm",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Apply LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 3. Test the Fine-Tuned Model (Inference)
def test_khattak_model(test_sentence):
    FastLanguageModel.for_inference(model)
    
    alpaca_prompt = """Below is an instruction. Write a detailed response in Pashto.

    ### Instruction:
    {}

    ### Response:
    """
    
    inputs = tokenizer(
    [
        alpaca_prompt.format(test_sentence, "")
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    
    print(f"English: {test_sentence}")
    print("Khattak AI:", response.split("### Response:\n")[-1])

# Example Test
test_khattak_model("Where are you going now?")
# Expected Output: ته اوس چےتا سې؟
