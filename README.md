# 🧭 Spatial Reasoning in VLMs (Qwen2.5-VL)

### 🎯 Objective

Enhance spatial reasoning capabilities in Vision-Language Models (VLMs) using **Chain-of-Thought (CoT)** prompting and reasoning control techniques inspired by **LMQL**. The goal is to improve reasoning accuracy in maze-based navigation tasks where the model must infer the agent's final position based on visual and action-sequence inputs.

---

### 🧠 Method Summary

* **Model:** `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit quantized)
* **Dataset:** 100 maze samples (`maze_clean_dataset/json`)
* **Task:** Predict which maze letter (A/B/C/D) the red agent reaches given an action sequence.
* **Baseline Prompt:** Direct question without reasoning.
* **Improved Prompt:** Structured CoT reasoning steps + low temperature sampling.

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

| Method                         | Accuracy           | Notes                                     |
| ------------------------------ | ------------------ | ----------------------------------------- |
| Naive Prompt                   | **0.280 (28/100)** | Simple Q&A style                          |
| CoT Prompt + no Temp. tuning   | **0.340 (34/100)** | Structured reasoning, more stable outputs |

💾 Results saved in:

* `eval_results/qwen2.5vl_maze_results.json`
* `eval_results/qwen2.5vl_maze_results_CoT.json`

---

### 🔗 Related Work

* **LLaVA / CLIP / Hugging Face:** provide strong multimodal backbones for visual-text alignment.
* **LMQL:** allows controlled, interpretable CoT and VoT prompting for stepwise reasoning — relevant for extending this pipeline to **3D spatial tasks**.

---

### 🚀 Next Steps

* Integrate **LMQL** for structured CoT/VoT control.
* Add temperature and top-p to decoding
* Experiment with **LLaVA** and **Qwen2-VL 7B** for higher reasoning fidelity.
* Explore **visual reasoning trace visualization** (Visualization-of-Thought).
* Extend dataset to **3D layouts** and **temporal sequences** for richer spatial grounding.
