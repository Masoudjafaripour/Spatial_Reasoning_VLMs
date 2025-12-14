# 🧭 Spatial Reasoning in VLMs (Qwen2.5-VL)

### 🎯 Objective

Enhance spatial reasoning capabilities in Vision-Language Models (VLMs) using **Chain-of-Thought (CoT)** prompting and reasoning control techniques inspired by **LMQL**. The goal is to improve reasoning accuracy in maze-based navigation tasks where the model must infer the agent's final position based on visual and action-sequence inputs.

---

### 🧠 Method Summary

* **Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit quantized)
* **Dataset:** 100 maze samples (`maze_clean_dataset/json`)
* **Task:** Predict which maze letter (A/B/C/D) the red agent reaches given an action sequence.
* **Baseline Prompt:** Direct question without reasoning.
* **CoT Prompting:** Structured step-by-step reasoning hints.
* **🆕 Prompt-RL (ICL-RL):** A lightweight **epsilon-greedy bandit** learns which prompt augmentation (e.g., action simulation, coordinate reasoning) maximizes task reward, **without updating model weights**.

---

### ⚙️ Inference Settings

| Setting        | Value     |
| -------------- | --------- |
| Quantization   | 4-bit NF4 |
| Max New Tokens | 50        |
| Temperature    | 0.2       |
| Top-p          | 0.9       |
| Device         | CUDA      |

---

### 📈 Results

| Method                                     | Accuracy           | Notes                                       |
| ------------------------------------------ | ------------------ | ------------------------------------------- |
| Naive Prompt                               | **0.280 (28/100)** | Simple Q&A style                            |
| CoT Prompt + no Temp. tuning               | **0.340 (34/100)** | Structured reasoning, more stable outputs   |
| Prompt-RL (In-Context RL over prompts)     | **0.290 (29/100)** | Bandit learns effective reasoning scaffolds |

💾 Results saved in:

* `eval_results/qwen2.5vl_maze_results.json`
* `eval_results/qwen2.5vl_maze_results_CoT.json`
* `eval_results/qwen2.5vl_maze_prompt_rl_results.json`

---

### 🧪 Prompt-RL Analysis

Learned prompt values after online interaction:

| Prompt Action        | Q-value   | Samples |
| -------------------- | --------- | ------- |
| simulate_actions     | **0.320** | 75      |
| base                 | 0.286     | 7       |
| coordinate_reasoning | 0.250     | 8       |
| step_by_step         | 0.125     | 8       |
| self_check           | 0.000     | 2       |

This shows that **explicit action simulation** is the most effective in-context reasoning scaffold for maze-based spatial reasoning, even without any parameter updates.

---

### 🔗 Related Work

* **LLaVA / CLIP / Hugging Face:** provide strong multimodal backbones for visual-text alignment.
* **LMQL:** enables controlled, interpretable CoT and VoT prompting for stepwise reasoning — relevant for extending this pipeline to **3D spatial tasks**.

---

### 🚀 Next Steps

* Combine **Prompt-RL with CoT** (multi-edit prompts per episode).
* Extend Prompt-RL to **state-aware RL** (condition on failure modes).
* Integrate **LMQL** for hard constraints over reasoning steps.
* Add temperature and top-p decoding control.
* Experiment with **LLaVA** and **Qwen2-VL 7B** for higher reasoning fidelity.
* Explore **Visualization-of-Thought** for spatial reasoning traces.
* Extend dataset to **3D layouts** and **temporal sequences**.
