# ============================================================
# Prompt-RL for Spatial Reasoning with Qwen2.5-VL
# ============================================================

import os, json, re, gc, random
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

# ---------------- CONFIG ----------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_DIR = "maze_clean_dataset/json"

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "maze_clean_dataset", "json")

MAX_NEW_TOKENS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPSILON = 0.2          # exploration rate for RL
SEED = 42
# ---------------------------------------

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- RL PROMPT ACTIONS ----------------
PROMPT_ACTIONS = {
    "base": "",
    "step_by_step": (
        "Think step by step and track the agent position carefully after each move.\n"
    ),
    "simulate_actions": (
        "Simulate each action sequentially and update the agent position explicitly.\n"
    ),
    "coordinate_reasoning": (
        "Reason using grid coordinates and relative directions (up/down/left/right).\n"
    ),
    "self_check": (
        "After reasoning, double-check the final agent position before answering.\n"
    ),
}

prompt_q = {k: 0.0 for k in PROMPT_ACTIONS}
prompt_n = {k: 0 for k in PROMPT_ACTIONS}


def select_prompt_action():
    if random.random() < EPSILON:
        return random.choice(list(PROMPT_ACTIONS.keys()))
    return max(prompt_q, key=lambda k: prompt_q[k])


def update_prompt_q(action, reward):
    prompt_n[action] += 1
    prompt_q[action] += (reward - prompt_q[action]) / prompt_n[action]


# ---------------- LOAD MODEL ----------------
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


# ---------------- INFERENCE ----------------
def get_prediction(image_path, action_seq, prompt_hint=""):
    image = Image.open(image_path).convert("RGB").resize((384, 384))

    prompt_text = (
        "You are an expert at spatial reasoning in mazes.\n"
        + prompt_hint +
        f"Action sequence: {action_seq}\n\n"
        "Which letter (A, B, C, or D) does the red agent reach?\n"
        "Answer only as: 'The answer is [A/B/C/D].'"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=False,
        )

    generated_ids = output[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del image, inputs, output, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    return text.strip()


def extract_letter(pred):
    m = re.search(r"\b([ABCD])\b", pred)
    return m.group(1) if m else None


# ---------------- EVALUATION + RL LOOP ----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, "maze_clean_dataset", "json")

files = sorted(
    [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".json")]
)

correct = total = 0
results = []

print(f"🔹 Evaluating {len(files)} maze samples with Prompt-RL...")

for path in tqdm(files):
    with open(path) as f:
        sample = json.load(f)

    image_path = sample["image"]
    gt = sample["answer"]
    actions = sample["actions"]

    # RL chooses prompt variant
    prompt_action = select_prompt_action()
    prompt_hint = PROMPT_ACTIONS[prompt_action]

    pred_text = get_prediction(image_path, actions, prompt_hint)
    pred_letter = extract_letter(pred_text)

    reward = 1.0 if pred_letter == gt else 0.0
    update_prompt_q(prompt_action, reward)

    results.append({
        "file": path,
        "gt": gt,
        "pred": pred_letter,
        "prompt_action": prompt_action,
        "reward": reward,
        "raw_output": pred_text,
    })

    if reward == 1.0:
        correct += 1
    total += 1


# ---------------- RESULTS ----------------
acc = correct / total if total else 0
print(f"\n✅ Accuracy: {acc:.3f} ({correct}/{total})")

print("\n📊 Learned prompt values:")
for k in sorted(prompt_q, key=lambda x: prompt_q[x], reverse=True):
    print(f"{k:22s} Q={prompt_q[k]:.3f} n={prompt_n[k]}")

os.makedirs("eval_results", exist_ok=True)
out_path = "eval_results/qwen2.5vl_maze_prompt_rl_results.json"
with open(out_path, "w") as f:
    json.dump(
        {
            "accuracy": acc,
            "prompt_q": prompt_q,
            "prompt_counts": prompt_n,
            "results": results,
        },
        f,
        indent=2,
    )

print(f"\n💾 Saved detailed results to {out_path}")
