import os, json, re, gc
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ---------------- CONFIG ----------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_DIR = "maze_clean_dataset/json"
MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

print("🔹 Loading model in 4-bit mode (low memory)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
torch.set_grad_enabled(False)

# ---------- Inference ----------
def get_prediction(image_path, action_seq):
    """Inference using Qwen2.5-VL with proper processor usage."""
    image = Image.open(image_path).convert("RGB").resize((384, 384))

    # Construct the prompt text
    # prompt_text = (
    #     "You are an expert at reasoning in mazes.\n"
    #     f"Action sequence: {action_seq}\n\n"
    #     "Which letter (A, B, C, or D) does the red agent reach?\n"
    #     "Answer only as: 'The answer is [A/B/C/D].'"
    # )

    prompt_text = (
        "You are a world-class spatial reasoning expert.\n"
        "Carefully analyze the maze image and reason step-by-step before deciding.\n"
        "Each action in the sequence corresponds to a movement of the red agent.\n\n"
        f"Action sequence: {action_seq}\n\n"
        "Please reason explicitly:\n"
        "- Step 1: Describe where the red agent starts.\n"
        "- Step 2: Follow each action and describe how its position changes.\n"
        "- Step 3: Identify the final position and match it to the nearest letter (A/B/C/D).\n"
        "Be concise but logical.\n\n"
        "Output exactly this format:\n"
        "Reasoning: <your reasoning>\n"
        "The answer is [A/B/C/D]."
    )

    # Construct conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply chat template to get text prompt
    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Process inputs with both image and text
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,      # ↓ makes reasoning more deterministic
            top_p=0.9,            # ↓ keeps coherent reasoning while allowing diversity
            use_cache=False,
        )

    # Decode only the generated part (remove input tokens)
    generated_ids = output[:, inputs['input_ids'].shape[1]:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    del image, inputs, output, generated_ids
    torch.cuda.empty_cache()
    gc.collect()
    return text.strip()


def extract_letter(pred):
    m = re.search(r"\b([ABCD])\b", pred)
    return m.group(1) if m else None


# ---------- Evaluation ----------
files = sorted(
    [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".json")]
)
correct = total = 0
results = []

print(f"🔹 Evaluating {len(files)} maze samples...")

for path in tqdm(files):
    with open(path) as f:
        sample = json.load(f)

    image_path, gt, actions = sample["image"], sample["answer"], sample["actions"]
    pred_text = get_prediction(image_path, actions)
    pred_letter = extract_letter(pred_text)

    results.append(
        {"file": path, "gt": gt, "pred": pred_letter, "raw_output": pred_text}
    )
    if pred_letter == gt:
        correct += 1
    total += 1

acc = correct / total if total else 0
print(f"\n✅ Accuracy: {acc:.3f} ({correct}/{total})")

os.makedirs("eval_results", exist_ok=True)
out_path = "eval_results/qwen2.5vl_maze_results_CoT.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"💾 Saved detailed results to {out_path}")

#  Accuracy: 0.340 (34/100) with adjustimg temperature and top_p
# 💾 Saved detailed results to eval_results/qwen2.5vl_maze_results_CoT.json