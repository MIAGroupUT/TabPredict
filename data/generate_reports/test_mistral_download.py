import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# CONFIG
# ----------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
CACHE_DIR = "/deepstore/datasets/mia/AI4EVAR/checkpoints/hf_models/qwen_7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# DOWNLOAD TOKENIZER & MODEL
# ----------------------------

print(f"Downloading {MODEL_NAME} to {CACHE_DIR} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,   # use fp16 for GPU
    device_map="auto",            # automatically map to GPU
    # attn_implementation="flash_attention_2"
)

print("Model loaded successfully!")

# ----------------------------
# TEST PROMPT
# ----------------------------
messages = [
    {"role": "system", "content": "You are a board-certified vascular radiologist."},
    {"role": "user", "content": "Write a short postoperative AAA surveillance radiology report for a patient with baseline aneurysm 55 mm and follow-up 50 mm. Use structured sections: INDICATION, TECHNIQUE, FINDINGS, IMPRESSION."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


# prompt = """
# You are a board-certified vascular radiologist.

# Write a short postoperative AAA surveillance radiology report
# for a patient with baseline aneurysm 55 mm and follow-up 50 mm.
# Use structured sections: INDICATION, TECHNIQUE, FINDINGS, IMPRESSION.
# """

# inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

print("Generating test report ...")
outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    temperature=0.1,
    do_sample=False
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n=== GENERATED REPORT ===\n")
print(text)
print("\n=== END OF REPORT ===\n")
print("\n=== GENERATED REPORT ===\n")
print(text)
print("\n=== END OF REPORT ===\n")