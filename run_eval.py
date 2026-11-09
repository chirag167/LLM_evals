"""
run_eval.py  (step 5: run every model under every knob setting)

Produces results.csv with ONE ROW per
    (model, template, n_shots, option_order, extraction, question)
holding a 0/1 "correct" column. analyze.py reads that file.

Layout of this file, top to bottom:
    1. Loading questions
    2. Building prompts         (knobs: template, n_shots, option_order)
    3. Extracting answers       (knob:  extraction)
    4. Models                   (a fake one for testing, a real one for later)
    5. The main loop

Run:  python run_eval.py config.json
"""

import json
import zlib
import re
import sys
import itertools
import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D"]


# =====================================================================
# 1. Loading questions
# =====================================================================
# Each question is a dict:
#   {"id": 17, "subject": "physics", "question": "...",
#    "choices": ["...", "...", "...", "..."], "answer": 1}
# "answer" is the index (0-3) of the correct choice.

def load_synthetic_questions(n, seed):
    """Made-up questions so the pipeline runs without downloading anything.
    The text is meaningless; only the fake model below can 'answer' them."""
    rng = np.random.default_rng(seed)
    subjects = [f"subject_{k}" for k in range(10)]
    test = []
    for i in range(n):
        test.append({"id": i,
                     "subject": subjects[i % 10],
                     "question": f"Synthetic question number {i}?",
                     "choices": [f"option {j} of q{i}" for j in range(4)],
                     "answer": int(rng.integers(4))})
    # 5 already-answered examples per subject, used for few-shot prompts
    dev = {s: [{"id": -1, "subject": s,
                "question": f"Example question {k} about {s}?",
                "choices": [f"choice {j}" for j in range(4)],
                "answer": int(rng.integers(4))} for k in range(5)]
           for s in subjects}
    return test, dev


def load_mmlu_questions(n, seed):
    """Real MMLU questions from Hugging Face. NOT TESTED in the sandbox
    where this was written (no internet access there). Needs:
        pip install datasets"""
    from datasets import load_dataset
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(test_ds), size=n, replace=False)
    test = [{"id": int(i), "subject": test_ds[int(i)]["subject"],
             "question": test_ds[int(i)]["question"],
             "choices": test_ds[int(i)]["choices"],
             "answer": int(test_ds[int(i)]["answer"])} for i in picks]
    dev = {}
    for row in dev_ds:
        dev.setdefault(row["subject"], []).append(
            {"id": -1, "subject": row["subject"], "question": row["question"],
             "choices": row["choices"], "answer": int(row["answer"])})
    return test, dev


# =====================================================================
# 2. Building prompts
# =====================================================================

def reorder_options(item, option_order, seed):
    """Knob 4. Returns (choices in display order, index of correct answer).
    'shuffled' uses a fixed permutation per question, so every model sees
    the SAME shuffle. That keeps the comparison paired."""
    if option_order == "original":
        return item["choices"], item["answer"]
    rng = np.random.default_rng(seed + item["id"] + 1000)
    perm = rng.permutation(4)                    # perm[new_pos] = old_pos
    choices = [item["choices"][k] for k in perm]
    correct = int(np.where(perm == item["answer"])[0][0])
    return choices, correct


def format_question(item, choices, template):
    """Knob 1. Writes one question in the chosen template wording."""
    if template == "plain":
        opts = " ".join(f"{L}) {c}" for L, c in zip(LETTERS, choices))
        return f"Question: {item['question']}\n{opts}\nAnswer:"
    if template == "instructed":
        opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))
        return (f"The following is a multiple-choice question about "
                f"{item['subject']}. Choose the correct option and reply "
                f"with its letter only.\n\n{item['question']}\n{opts}\nAnswer:")
    raise ValueError(f"unknown template {template}")


def build_few_shot(item, dev, template, n_shots):
    """Knob 2. Puts n_shots answered examples (same subject, same template)
    in front of the real question. Returns a list of example strings
    (empty when n_shots is 0)."""
    parts = []
    for ex in dev[item["subject"]][:n_shots]:
        ex_text = format_question(ex, ex["choices"], template)
        parts.append(f"{ex_text} {LETTERS[ex['answer']]}")
    return parts


# =====================================================================
# 3. Extracting answers  (knob 5: grading, applied AFTER the model call)
# =====================================================================
# Each function returns "A"/"B"/"C"/"D", or None if no answer is found.
# None is graded as wrong.

def extract_first_letter(reply, probs):
    """Strict: the reply must start with the letter."""
    text = reply.strip()
    if text and text[0].upper() in LETTERS:
        return text[0].upper()
    return None


def extract_regex(reply, probs):
    """Lenient: the first standalone A/B/C/D anywhere in the reply.
    Known flaw: the English article 'A' ('A newton is...') counts too."""
    m = re.search(r"\b([ABCD])\b", reply)
    return m.group(1) if m else None


def extract_logprob(reply, probs):
    """Ignores the text. Picks the letter the model gave the highest
    probability right after 'Answer:'. Needs an open model."""
    return max(probs, key=probs.get)


EXTRACTORS = {"first_letter": extract_first_letter,
              "regex": extract_regex,
              "logprob": extract_logprob}


# =====================================================================
# 4. Models
# =====================================================================
# Every model is used through one function call:
#     reply_text, letter_probs = ask(model, prompt, correct_letter)
# letter_probs is a dict like {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}.
# Only the FAKE model looks at correct_letter. A real model ignores it.

def make_fake_model(name):
    """A pretend model with built-in quirks, so the analysis has real
    systematic effects to find. Every number here is invented."""
    quirks = {
        "model_A": {"skill": 0.55, "instructed_bonus": 0.10,
                    "bias_toward_A": 0.00, "verbose_when_0shot": False},
        "model_B": {"skill": 0.45, "instructed_bonus": 0.35,
                    "bias_toward_A": 0.40, "verbose_when_0shot": True},
    }
    return {"kind": "fake", "name": name, **quirks[name]}


def ask_fake(model, prompt, correct_letter, item_id, template, n_shots):
    """Fake model: scores each option, picks the best one (greedy)."""
    rng = np.random.default_rng(item_id)       # same difficulty for every model
    difficulty = rng.normal(0, 1.2)
    noise = np.random.default_rng(zlib.crc32(f"{model['name']}-{item_id}".encode()))
    scores = noise.normal(0, 0.8, size=4)      # model-specific question quirks
    k = LETTERS.index(correct_letter)
    scores[k] += 1.5 * (model["skill"] - 0.25 * difficulty)
    if template == "instructed":
        scores[k] += model["instructed_bonus"]
    if n_shots > 0:
        scores[k] += 0.15
    scores[0] += model["bias_toward_A"]        # option position bias
    probs = np.exp(scores) / np.exp(scores).sum()
    probs = dict(zip(LETTERS, probs))
    pick = max(probs, key=probs.get)
    if model["verbose_when_0shot"] and n_shots == 0:
        reply = f"The answer is ({pick})."
    else:
        reply = pick
    return reply, probs


def make_hf_model(name):
    """Real open-weight model from Hugging Face, e.g.
    'Qwen/Qwen2.5-0.5B-Instruct'. NOT TESTED in the sandbox where this was
    written. Needs:  pip install torch transformers accelerate"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    net = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto",
                                               device_map="auto")
    # Token ids for " A", " B", ... (with a leading space, as they usually
    # appear after "Answer:"). Check these for your model's tokenizer.
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[-1]
                  for L in LETTERS]
    return {"kind": "hf", "name": name, "tok": tok, "net": net,
            "letter_ids": letter_ids, "torch": torch}


def ask_hf(model, prompt, max_new_tokens=15):
    """One forward pass for the letter probabilities, then a short greedy
    generation for the reply text."""
    torch, tok, net = model["torch"], model["tok"], model["net"]
    inputs = tok(prompt, return_tensors="pt").to(net.device)
    with torch.no_grad():
        logits = net(**inputs).logits[0, -1]
        letter_logits = logits[model["letter_ids"]].float()
        p = torch.softmax(letter_logits, dim=0).cpu().numpy()
        out = net.generate(**inputs, max_new_tokens=max_new_tokens,
                           do_sample=False)
    reply = tok.decode(out[0, inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True)
    return reply, dict(zip(LETTERS, p))


# =====================================================================
# 5. Main loop
# =====================================================================

def run(config):
    seed = config["seed"]
    if config["dataset"] == "synthetic":
        test, dev = load_synthetic_questions(config["n_questions"], seed)
    else:
        test, dev = load_mmlu_questions(config["n_questions"], seed)

    knobs = config["knobs"]
    rows = []
    for model_name in config["models"]:
        kind, name = model_name.split(":", 1) if ":" in model_name \
                     else ("hf", model_name)
        model = make_fake_model(name) if kind == "fake" else make_hf_model(name)

        # The model is called once per (template, n_shots, option_order).
        # Extraction is applied afterwards to the same reply: no extra calls.
        for template, n_shots, order in itertools.product(
                knobs["template"], knobs["n_shots"], knobs["option_order"]):
            print(f"{name:<12} template={template:<10} shots={n_shots} "
                  f"order={order}")
            for item in test:
                choices, correct_idx = reorder_options(item, order, seed)
                correct_letter = LETTERS[correct_idx]
                shots = build_few_shot(item, dev, template, n_shots)
                question = format_question(item, choices, template)
                prompt = "\n\n".join(shots + [question])

                if model["kind"] == "fake":
                    reply, probs = ask_fake(model, prompt, correct_letter,
                                            item["id"], template, n_shots)
                else:
                    reply, probs = ask_hf(model, prompt)

                for ext_name in knobs["extraction"]:
                    picked = EXTRACTORS[ext_name](reply, probs)
                    rows.append({"model": name, "template": template,
                                 "n_shots": n_shots, "option_order": order,
                                 "extraction": ext_name,
                                 "question_id": item["id"],
                                 "subject": item["subject"],
                                 "correct": int(picked == correct_letter),
                                 "p_correct": float(probs[correct_letter])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(config_path) as f:
        config = json.load(f)
    df = run(config)
    df.to_csv("results.csv", index=False)
    print(f"\nWrote results.csv with {len(df):,} rows")
