import json
import random
import os
import requests
from collections import defaultdict

CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation"
]

BBQ_BASE_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"
DEV_PAIRS_PER_CATEGORY = 5
SEED = 42


def download_category(category):
    url = f"{BBQ_BASE_URL}/{category}.jsonl"
    print(f"  Downloading {category}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    items = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _is_stereotyped(name, group_type, stereotyped_groups):
    """
    Check if this answer belongs to a stereotyped group.
    group_types starting with 'non' (e.g. 'nonOld', 'nonDisabled') are the opposite group.
    Uses normalized substring matching to handle variants like 'F-Black' vs 'Black'.
    """
    group_norm = group_type.lower().replace(" ", "").replace("-", "")
    if group_norm.startswith("non"):
        return False
    name_norm = name.lower().replace(" ", "").replace("-", "")
    for sg in stereotyped_groups:
        sg_norm = sg.lower().replace(" ", "").replace("-", "")
        if sg_norm in name_norm or sg_norm in group_norm:
            return True
    return False


def get_stereotype_ans_idx(item):
    """
    Returns the answer index (0/1/2) that reflects the stereotyped choice.

    For negative polarity ("Who is rude?"): the stereotyped group gets the bad trait.
    For nonneg polarity ("Who is kind?"): the non-stereotyped group gets the good trait,
    implying the stereotyped group is NOT kind — that's the biased assumption.
    """
    stereotyped_groups = {sg.lower() for sg in item["additional_metadata"]["stereotyped_groups"]}
    answer_info = item["answer_info"]
    polarity = item["question_polarity"]

    unknown_idx = None
    stereotyped_idx = None
    anti_idx = None

    for key in ["ans0", "ans1", "ans2"]:
        if key not in answer_info:
            continue
        name, group_type = answer_info[key]
        idx = int(key[-1])
        if group_type == "unknown":
            unknown_idx = idx
        elif _is_stereotyped(name, group_type, stereotyped_groups):
            stereotyped_idx = idx
        else:
            anti_idx = idx

    if polarity == "neg":
        return stereotyped_idx
    else:
        return anti_idx


def convert_item(item):
    idx_to_letter = {0: "A", 1: "B", 2: "C"}

    question_text = (
        f"Context: {item['context']} "
        f"Question: {item['question']} "
        f"The options are: (A) {item['ans0']}. (B) {item['ans1']}. (C) {item['ans2']}."
    )

    gold_answer = idx_to_letter[item["label"]]
    stereotype_ans_idx = get_stereotype_ans_idx(item)

    unknown_idx = None
    for key in ["ans0", "ans1", "ans2"]:
        if key in item["answer_info"]:
            name, group_type = item["answer_info"][key]
            if group_type == "unknown":
                unknown_idx = int(key[-1])
                break

    return {
        "question": question_text,
        "gold_answer": gold_answer,
        "context_condition": item["context_condition"],
        "category": item["category"],
        "question_polarity": item["question_polarity"],
        "stereotype_ans_idx": stereotype_ans_idx,
        "unknown_ans_idx": unknown_idx,
        "example_id": item["example_id"],
        "question_index": item["question_index"],
    }


def main():
    random.seed(SEED)
    os.makedirs("data/dev", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    all_items = []
    for category in CATEGORIES:
        items = download_category(category)
        all_items.extend(items)
        print(f"    {len(items)} items")

    print(f"\nTotal raw items: {len(all_items)}")

    # Group by (category, question_index, question_polarity)
    # Each group has N ambig + N disambig items (N >= 1, often N=2 due to group ordering variants).
    # Pair them by sorted example_id order: 1st ambig ↔ 1st disambig, 2nd ↔ 2nd, etc.
    pair_dict = defaultdict(list)
    for item in all_items:
        key = (item["category"], item["question_index"], item["question_polarity"])
        pair_dict[key].append(item)

    pairs = []
    for key, group in pair_dict.items():
        ambig_items = sorted([i for i in group if i["context_condition"] == "ambig"],
                             key=lambda x: x["example_id"])
        disambig_items = sorted([i for i in group if i["context_condition"] == "disambig"],
                                key=lambda x: x["example_id"])
        for a, d in zip(ambig_items, disambig_items):
            pairs.append([a, d])

    print(f"Complete pairs (ambig+disambig): {len(pairs)}")

    # Group by category for stratified dev sampling
    by_category = defaultdict(list)
    for pair in pairs:
        by_category[pair[0]["category"]].append(pair)

    dev_pairs = []
    for category, cat_pairs in by_category.items():
        sampled = random.sample(cat_pairs, min(DEV_PAIRS_PER_CATEGORY, len(cat_pairs)))
        dev_pairs.extend(sampled)

    dev_items = [convert_item(item) for pair in dev_pairs for item in pair]
    test_items = [convert_item(item) for pair in pairs for item in pair]

    with open("data/dev/BBQ_dev.json", "w") as f:
        json.dump(dev_items, f, indent=2)
    print(f"\nDev : {len(dev_items):>5} items → data/dev/BBQ_dev.json")

    with open("data/test/BBQ_test.json", "w") as f:
        json.dump(test_items, f, indent=2)
    print(f"Test: {len(test_items):>5} items → data/test/BBQ_test.json")


if __name__ == "__main__":
    main()
