import os
import numpy as np
import json
import random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from typing import List

def get_seed_list() -> List[int]:
    """Read seeds from ../assets/seed_list.txt (one integer per line)."""
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


# Load API key from disk (expects a file named 'api_key.txt' in the same directory)
def load_api_key(#
        path: str = "../keys/openai_api_key.txt"
) -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found at: {path}")
    return key_path.read_text().strip()


def make_context(respondent: str, target: str) -> str:
    """
    respondent: 'rep' or 'dem'
    target:     'rep' or 'dem'
    """
    if respondent == "rep":
        who = (
            "ideologically conservative. "
            "Politically, they are a strong Republican. "
            "Racially, they are white. "
            "They are male. "
            "Financially, they are upper-middle class. "
            "In terms of age, they are young."
        )
    elif respondent == "dem":
        who = (
            " ideologically liberal. "
            "Politically, they are a strong Democrat. "
            "Racially, they are white. "
            "They are female. "
            "Financially, they are poor. "
            "In terms of age, they are old."
        )
    else:
        raise ValueError(f"Unknown respondent {respondent!r}")

    if target == "rep":
        target_str = "Republican"
    elif target == "dem":
        target_str = "Democratic"
    else:
        raise ValueError(f"Unknown target {target!r}")

    return (
        f"The respondent is {who}"
        f"They list words describing {target_str} voters. "
        "They give exactly the following words."
    )


def main():
    import csv
    import json
    import random
    from tqdm import tqdm

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini-2024-07-18"

    # ---------------------------------------------------------
    # stereotype lists
    # ---------------------------------------------------------

    # Republican-provided list describing Democrats
    allowed_words_dem = [
        "liberal", "young", "progressive", "poor", "open-minded", "female",
        "educated", "intelligent", "pro-choice", "compassionate", "caring",
        "socialist", "communist", "athiest",
    ]

    # Democrat-provided list describing Republicans
    allowed_words_rep = [
        "conservative", "white", "religious", "racist", "rich",
        "old", "christian", "male", "homophobic", "patriotic",
        "ignorant", "traditional", "selfish", "responsible",
    ]

    # ---------------------------------------------------------
    # system prompt enforcing 1-word JSON output
    # ---------------------------------------------------------

    system_prompt = (
        "You are given a list of words that a respondent uses to describe "
        "supporters of a political group.\n\n"
        "Your task:\n"
        "1. Choose exactly ONE word from that list.\n"
        "2. The chosen word MUST be exactly one of the words from that list.\n"
        "3. Output exactly a JSON object of the form {\"label\": \"<word>\"}.\n"
        "4. Output NOTHING else."
    )

    # ---------------------------------------------------------
    # sampling parameters
    # ---------------------------------------------------------

    temperatures = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    top_p = 0.95
    seeds = get_seed_list()[0:1000]

    rows = []

    for temperature in tqdm(temperatures, desc="temperatures"):
        for seed in tqdm(seeds, desc="seeds", leave=False):
            rng = random.Random(seed)

            # ===============================================================
            # 1) Republican describing Democrats (rep -> dem)
            # ===============================================================
            shuffled_dem_1 = allowed_words_dem[:]
            rng.shuffle(shuffled_dem_1)

            ctx_1 = make_context("rep", "dem")
            prompt_1 = f"{ctx_1}\n\nWords: {', '.join(shuffled_dem_1)}"

            resp_1 = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                frequency_penalty=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_1},
                ],
            )

            try:
                label_1 = json.loads(resp_1.choices[0].message.content)["label"]
            except Exception:
                label_1 = allowed_words_dem[0]
            if label_1 not in allowed_words_dem:
                label_1 = allowed_words_dem[0]

            # ===============================================================
            # 2) Democrat describing Republicans (dem -> rep)
            # ===============================================================
            shuffled_rep_2 = allowed_words_rep[:]
            rng.shuffle(shuffled_rep_2)

            ctx_2 = make_context("dem", "rep")
            prompt_2 = f"{ctx_2}\n\nWords: {', '.join(shuffled_rep_2)}"

            resp_2 = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                frequency_penalty=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_2},
                ],
            )

            try:
                label_2 = json.loads(resp_2.choices[0].message.content)["label"]
            except Exception:
                label_2 = allowed_words_rep[0]
            if label_2 not in allowed_words_rep:
                label_2 = allowed_words_rep[0]

            # ===============================================================
            # 3) Democrat describing Democrats (dem -> dem)
            # ===============================================================
            shuffled_dem_3 = allowed_words_dem[:]
            rng.shuffle(shuffled_dem_3)

            ctx_3 = make_context("dem", "dem")
            prompt_3 = f"{ctx_3}\n\nWords: {', '.join(shuffled_dem_3)}"

            resp_3 = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                frequency_penalty=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_3},
                ],
            )

            try:
                label_3 = json.loads(resp_3.choices[0].message.content)["label"]
            except Exception:
                label_3 = allowed_words_dem[0]
            if label_3 not in allowed_words_dem:
                label_3 = allowed_words_dem[0]

            # ===============================================================
            # 4) Republican describing Republicans (rep -> rep)
            # ===============================================================
            shuffled_rep_4 = allowed_words_rep[:]
            rng.shuffle(shuffled_rep_4)

            ctx_4 = make_context("rep", "rep")
            prompt_4 = f"{ctx_4}\n\nWords: {', '.join(shuffled_rep_4)}"

            resp_4 = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                frequency_penalty=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_4},
                ],
            )

            try:
                label_4 = json.loads(resp_4.choices[0].message.content)["label"]
            except Exception:
                label_4 = allowed_words_rep[0]
            if label_4 not in allowed_words_rep:
                label_4 = allowed_words_rep[0]

            # ===============================================================
            # store compact row (no raw JSON, no prompt_order)
            # ===============================================================

            rows.append({
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,

                "rep_to_dem_label": label_1,
                "dem_to_rep_label": label_2,
                "dem_to_dem_label": label_3,
                "rep_to_rep_label": label_4,
            })

    # ---------------------------------------------------------
    # write CSV
    # ---------------------------------------------------------

    if rows:
        fieldnames = list(rows[0].keys())
        with open("results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ----------------------------------------------------------------------
# NEW FUNCTION: 1000 calls with SAME seed and temperature = 0,
# saved to a separate CSV.
# ----------------------------------------------------------------------
def run_fixed_seed_temp0_experiment():
    import csv
    import json
    import random

    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    model_name = "gpt-4o-mini-2024-07-18"

    # stereotype lists (same as in main)
    allowed_words_dem = [
        "liberal", "young", "progressive", "poor", "open-minded", "female",
        "educated", "intelligent", "pro-choice", "compassionate", "caring",
        "socialist", "communist", "athiest",
    ]

    allowed_words_rep = [
        "conservative", "white", "religious", "racist", "rich",
        "old", "christian", "male", "homophobic", "patriotic",
        "ignorant", "traditional", "selfish", "responsible",
    ]

    system_prompt = (
        "You are given a list of words that a respondent uses to describe "
        "supporters of a political group.\n\n"
        "Your task:\n"
        "1. Choose exactly ONE word from that list.\n"
        "2. The chosen word MUST be exactly one of the words from that list.\n"
        "3. Output exactly a JSON object of the form {\"label\": \"<word>\"}.\n"
        "4. Output NOTHING else."
    )

    temperature = 0.0
    top_p = 0.95
    runs = 1000

    # use a single fixed seed for all 1000 calls
    seed = get_seed_list()[0]

    rows = []

    for run_id in range(runs):
        rng = random.Random(seed)

        # 1) Republican describing Democrats (rep -> dem)
        shuffled_dem_1 = allowed_words_dem[:]
        rng.shuffle(shuffled_dem_1)

        ctx_1 = make_context("rep", "dem")
        prompt_1 = f"{ctx_1}\n\nWords: {', '.join(shuffled_dem_1)}"

        resp_1 = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            frequency_penalty=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_1},
            ],
        )

        try:
            label_1 = json.loads(resp_1.choices[0].message.content)["label"]
        except Exception:
            label_1 = allowed_words_dem[0]
        if label_1 not in allowed_words_dem:
            label_1 = allowed_words_dem[0]

        # 2) Democrat describing Republicans (dem -> rep)
        shuffled_rep_2 = allowed_words_rep[:]
        rng.shuffle(shuffled_rep_2)

        ctx_2 = make_context("dem", "rep")
        prompt_2 = f"{ctx_2}\n\nWords: {', '.join(shuffled_rep_2)}"

        resp_2 = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            frequency_penalty=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_2},
            ],
        )

        try:
            label_2 = json.loads(resp_2.choices[0].message.content)["label"]
        except Exception:
            label_2 = allowed_words_rep[0]
        if label_2 not in allowed_words_rep:
            label_2 = allowed_words_rep[0]

        # 3) Democrat describing Democrats (dem -> dem)
        shuffled_dem_3 = allowed_words_dem[:]
        rng.shuffle(shuffled_dem_3)

        ctx_3 = make_context("dem", "dem")
        prompt_3 = f"{ctx_3}\n\nWords: {', '.join(shuffled_dem_3)}"

        resp_3 = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            frequency_penalty=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_3},
            ],
        )

        try:
            label_3 = json.loads(resp_3.choices[0].message.content)["label"]
        except Exception:
            label_3 = allowed_words_dem[0]
        if label_3 not in allowed_words_dem:
            label_3 = allowed_words_dem[0]

        # 4) Republican describing Republicans (rep -> rep)
        shuffled_rep_4 = allowed_words_rep[:]
        rng.shuffle(shuffled_rep_4)

        ctx_4 = make_context("rep", "rep")
        prompt_4 = f"{ctx_4}\n\nWords: {', '.join(shuffled_rep_4)}"

        resp_4 = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            frequency_penalty=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_4},
            ],
        )

        try:
            label_4 = json.loads(resp_4.choices[0].message.content)["label"]
        except Exception:
            label_4 = allowed_words_rep[0]
        if label_4 not in allowed_words_rep:
            label_4 = allowed_words_rep[0]

        rows.append({
            "run_id": run_id,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "rep_to_dem_label": label_1,
            "dem_to_rep_label": label_2,
            "dem_to_dem_label": label_3,
            "rep_to_rep_label": label_4,
        })

    if rows:
        fieldnames = list(rows[0].keys())
        with open("results_fixed_seed_temp0.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
    run_fixed_seed_temp0_experiment()
