import torch
import json
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

############################################
# ARGUMENTS
############################################

parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)
parser.add_argument("--patient", type=str, default=None)
parser.add_argument("--output", default="/deepstore/datasets/mia/AI4EVAR/clinical_data/reports.csv")
# Updated default model and cache dir for Qwen
parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--cachedir", default="/deepstore/datasets/mia/AI4EVAR/checkpoints/hf_models/qwen_7b")

args = parser.parse_args()

############################################
# LOAD MODEL
############################################

print("Loading model:", args.model)

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cachedir)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    cache_dir=args.cachedir,
    torch_dtype=torch.float16,   
    device_map="auto",            
    # attn_implementation="flash_attention_2" # Enabled for L40s
)

############################################
# PROMPT
############################################

# - "AAAdiam" is an Aortic Aneurysm (dilation), NOT a stenosis (narrowing).

SYSTEM_PROMPT = """You are a board-certified vascular radiologist.
Generate a radiology report using the provided clinical dataset.

Requirements:
- Use formal radiology terminology.
- Sections: INDICATION, TECHNIQUE, FINDINGS, COMPARISON, IMPRESSION.
- Use ONLY provided data. Do NOT invent values, history, or symptoms.
- Never change the user/patient ID, you should always keep it as is.

Output MUST be a single valid JSON object:
{
 "patient_id": "...",
 "report": {"indication": "...", "technique": "...", "findings": "...", "comparison": "...", "impression": "..."},
 "evidence": [{"variable": "...", "value": "...", "statement": "..."}]
}"""

############################################
# LOAD DATA
############################################

with open(args.input) as f:
    patients = json.load(f)

mapping_path = args.input.replace(".json", "_mapping.json")
try:
    with open(mapping_path) as f:
        variable_mapping = json.load(f)
    print("Variable mapping loaded.")
except FileNotFoundError:
    variable_mapping = {}
    print("Warning: No mapping file found!")

############################################
# GENERATE REPORT
############################################

def generate_report(patient, mapping):
    # Use Chat Template for Qwen to distinguish instructions from data
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DATA DICTIONARY:\n{json.dumps(mapping)}\n\nPATIENT DATA:\n{json.dumps(patient)}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=False
        )

    # Decode only the response part
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(full_text)
    
    # Extract JSON block
    # try:
    #     json_start = full_text.rfind('{')
    #     json_end = full_text.rfind('}') + 1
    #     clean_json = full_text[json_start:json_end]
    #     result = json.loads(clean_json)
    # except:
    result = {"error": "parsing_failed", "raw": full_text}

    return result

############################################
# MAIN LOOP
############################################

reports = []
target_id = args.patient

if target_id:
    if target_id in patients:
        to_process = [(target_id, patients[target_id])]
    else:
        print(f"Patient {target_id} not found.")
        to_process = []
else:
    to_process = patients.items()

for pid, p_data in to_process:
    print(f"Generating report for {pid}...")
    report = generate_report(p_data, variable_mapping)
    reports.append({"PatientNr": pid, "report": report})

############################################
# SAVE
############################################

df = pd.DataFrame(reports)
df.to_csv(args.output, index=False)
print(f"Saved to {args.output}")
