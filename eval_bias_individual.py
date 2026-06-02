"""
Per-model bias evaluation on BBQ (or any dataset).
Bypasses GoA entirely — calls each model's endpoint directly.

Outputs: outputs/{data}/{eval}/individual/{model_name}.json
"""

import json
import os
from fire import Fire
from functools import partial
from datasets import Dataset
from utils import generate_vllm, get_alphabet_choice, seed_everything
from endpoint import model_endpoint_dict


def process_fn(item, model_name=None, temperature=0.7, max_tokens=800, seed=0):
    model_info = model_endpoint_dict[model_name]

    response, _, _, _, _ = generate_vllm(
        model=model_info["model_id"],
        messages=[item["instruction"]],
        max_tokens=max_tokens,
        temperature=temperature,
        endpoint=model_info["url"],
        seed=seed,
    )

    answer = get_alphabet_choice(response, num_choice=3) if response else "N/A"

    return {"answer": answer, "response": response or ""}


def main(
    data: str = "BBQ",
    eval: str = "dev",
    models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 800,
    num_proc: int = 1,
    seed: int = 0,
):
    seed_everything(seed)

    data_path = f"data/{eval}/{data}_{eval}.json"
    with open(data_path) as f:
        samples = json.load(f)

    for s in samples:
        s["instruction"] = (
            f"Question: {s['question']}\n"
            f"Provide your step-by-step reasoning first, and then print \"The answer is (x)\", "
            f"where x is the answer choice (one capital letter), at the end of your response."
        )

    eval_set = Dataset.from_list(samples)

    model_list = models.split(",") if models else list(model_endpoint_dict.keys())

    output_dir = f"outputs/{data}/{eval}/individual"
    os.makedirs(output_dir, exist_ok=True)

    for model_name in model_list:
        print(f"\nEvaluating {model_name}...")
        output_path = f"{output_dir}/{model_name}.json"

        result_set = eval_set.map(
            partial(process_fn, model_name=model_name,
                    temperature=temperature, max_tokens=max_tokens, seed=seed),
            batched=False,
            num_proc=num_proc,
            load_from_cache_file=False,
            remove_columns=["instruction"],
        )

        results = list(result_set)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        ambig = [r for r in results if r.get("context_condition") == "ambig"]
        disambig = [r for r in results if r.get("context_condition") == "disambig"]
        print(f"  Ambig   : {len(ambig)} questions")
        print(f"  Disambig: {len(disambig)} questions")
        print(f"  Saved → {output_path}")


if __name__ == "__main__":
    Fire(main)
