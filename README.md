# Khatta-ka AI: Khattak Dialect Pashto LLM 🏔️

[![Live Demo](https://img.shields.io/badge/Live_Demo-Try_Khattak_AI-orange?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Muhammad-Zubair796/Khattak-AI-Demo)
[![Portfolio](https://img.shields.io/badge/Case_Study-View_on_Portfolio-emerald?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://www.mzubair.online/projects/khatta-ka-llm)
[![Model on HF](https://img.shields.io/badge/Hugging_Face-View_Model-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Muhammad-Zubair796/Khatta-ka)

The world's first AI language model fine-tuned by Muhammad Zubair  specifically to understand and generate the **Khattak dialect** of the Pashto language. 

Standard Pashto AI models often fail to capture the rich, localized grammar and vocabulary of rural dialects. This project bridges that gap by fine-tuning a base Pashto LLM to speak exactly like a native from the Khattak tribe regions (Karak, Nowshera, Kohat).

## 📸 Model in Action
*(Here is the AI successfully translating English into pure Khattak Pashto on unseen test sentences)*

![Khattak AI Output 2](khatta-ka3.PNG)

## 🚀 Why This Matters
Standard Pashto models use Peshawari or Kandahari dialects. This model was specifically trained to recognize unique Khattak linguistic markers:
* **Future Tense:** Uses `بو` (bo) instead of standard `به` (ba). *(e.g., زه بو سبو چار کاوں)*
* **Possession:** Uses `مو والا` (mo wala) instead of standard `زما` (zama).
* **Vocabulary:** Uses pure Khattak words like `استر` (astr - big), `کسو` (kso - bad/dirty), and `سبو` (sabo - tomorrow).
* **Gendered Adjectives:** Perfectly adapts feminine/masculine rules unique to the dialect (e.g., `ستره` vs `استر`).

## 🛠️ How It Was Built
* **Base Model:** `junaid008/qehwa-pashto-llm` (Qwen2 architecture)
* **Framework:** [Unsloth](https://github.com/unslothai/unsloth) for 2x faster LoRA fine-tuning.
* **Dataset:** A custom-built, highly curated dataset of 2,000 English-to-Khattak sentence pairs focusing on everyday conversation, grammar rules, and local idioms.
* **Hardware:** Trained on a single Tesla T4 GPU via Google Colab.

## 📈 Training Performance
The model was trained for 4 epochs (620 steps). The training loss steadily and successfully decreased from **3.44** down to **0.22**, indicating excellent adaptation to the Khattak dataset without overfitting.

![Training Process](khatta-ka2.PNG)

<details>
<summary><b>Click here to view the full Training Loss History</b></summary>
Step	Training Loss
10	3.447874
20	2.150129
30	1.407591
40	1.039266
50	0.920436
60	0.834459
70	0.821306
80	0.715990
90	0.718809
100	0.638106
110	0.604851
120	0.559561
130	0.577652
140	0.525185
150	0.497480
160	0.513509
170	0.400574
180	0.392234
190	0.430440
200	0.397762
210	0.416910
220	0.399505
230	0.420360
240	0.382605
250	0.393253
260	0.389418
270	0.429069
280	0.356388
290	0.377030
300	0.398721
310	0.377821
320	0.273812
330	0.269704
340	0.276601
350	0.283776
360	0.305166
370	0.272352
380	0.310798
390	0.280581
400	0.276446
410	0.297413
420	0.286290
430	0.282724
440	0.306337
450	0.285896
460	0.278556
470	0.269478
480	0.245348
490	0.229646
500	0.228311
510	0.221181
520	0.246099
530	0.227797
540	0.233362
550	0.222423
560	0.222264
570	0.226506
580	0.230401
590	0.227558
600	0.233159
610	0.228642
620	0.224542
</details>

## 💻 How to Use the Model

### Option 1: Live Demo (No Coding)
Test the model directly in your browser here: **[Khattak AI Live Demo](https://huggingface.co/spaces/Muhammad-Zubair796/Khattak-AI-Demo)**

### Option 2: Python Script
You can load this model directly from Hugging Face using Unsloth:

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Muhammad-Zubair796/Khatta-ka",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Test Sentence
inputs = tokenizer(["English: I will go to the big market tomorrow.\nKhattak:"], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 64)
print(tokenizer.batch_decode(outputs))
