"""
TunedAI Labs — Raw Passage Reasoning Test
==========================================
Tests the fine-tuned model against 6 verbatim pre-AI passages.
Runs both Base Qwen and TunedAI Labs model, scores each answer,
saves full results to raw_passage_results.json.

Usage:
  pip install transformers peft accelerate bitsandbytes torch
  python raw_passage_test.py

  # On RunPod / Colab (full precision):
  python raw_passage_test.py

  # On local Mac / limited VRAM (4-bit quantization):
  python raw_passage_test.py --quantize
"""

import argparse
import json
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "tunedailabs/causal-reasoning-qwen-7b"
OUTPUT_FILE  = "raw_passage_results.json"
MAX_TOKENS   = 400

SYSTEM = (
    "You are a careful reasoner. Answer questions about causation, association, "
    "intervention, and counterfactuals precisely and correctly."
)

# ── Test Questions ─────────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": 1,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part II §59, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"The first time a man saw the communication of motion by impulse, as by the shock of two '
            'billiard balls, he could not pronounce that the one event was connected: but only that it '
            'was conjoined with the other. After he has observed several instances of this nature, he '
            'then pronounces them to be connected. What alteration has happened to give rise to this new '
            'idea of connexion? Nothing but that he now feels these events to be connected in his '
            'imagination, and can readily foretell the existence of one from the appearance of the other. '
            'When we say, therefore, that one object is connected with another, we mean only that they '
            'have acquired a connexion in our thought, and give rise to this inference, by which they '
            'become proofs of each other\'s existence."'
        ),
        "question": (
            "According to this passage, what can a man conclude the FIRST time he sees one billiard ball "
            "strike another? After observing several such instances, Hume says he pronounces them "
            '"connected." What does Hume say has actually changed — and what has NOT changed in the '
            "physical world?"
        ),
        "correct_answer": (
            "The first time, the man can only say the events were conjoined — not connected. "
            "After repetition, a feeling of connexion forms in his imagination (habit/custom). "
            "What changed is purely mental. Nothing changed in the physical world — the connexion "
            "exists in the mind, not in the objects themselves."
        ),
        "score_keywords": [
            ["conjoin", "conjoined"],
            ["imagination", "habit", "custom", "thought", "mind"],
            ["nothing", "no change", "unchanged", "physical world", "objects"],
        ],
    },
    {
        "id": 2,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part II §60, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"Similar objects are always conjoined with similar. Of this we have experience. Suitably to '
            'this experience, therefore, we may define a cause to be an object, followed by another, and '
            'where all the objects similar to the first are followed by objects similar to the second. '
            'Or in other words where, if the first object had not been, the second never had existed."'
        ),
        "question": (
            "Hume gives two definitions of a cause in this passage. State both definitions in your own "
            "words. According to his second definition, what question would you ask to test whether "
            "event A caused event B?"
        ),
        "correct_answer": (
            "First definition: constant conjunction — a cause is an object always followed by another "
            "of a similar kind (regularity theory). Second definition: counterfactual — if the first "
            "object had not existed, the second would never have existed. To test: ask 'If A had not "
            "occurred, would B still have occurred?' If no, A caused B."
        ),
        "score_keywords": [
            ["constant conjunction", "always followed", "regularity", "similar objects"],
            ["counterfactual", "had not been", "never had existed", "if...not"],
            ["test", "ask", "question", "would B", "would it"],
        ],
    },
    {
        "id": 3,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd edition, Chapter IX, 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"As there is no difference whatever, either in the houses or the people receiving the supply '
            'of the two Water Companies, or in any of the physical conditions with which they are '
            'surrounded, it is obvious that no experiment could have been devised which would more '
            'thoroughly test the effect of water supply on the progress of cholera than this, which '
            'circumstances placed ready made before the observer. The experiment, too, was on the '
            'grandest scale. No fewer than three hundred thousand people of both sexes, of every age '
            'and occupation, and of every rank and station, from gentlefolks down to the very poor, '
            'were divided into two groups without their choice, and, in most cases, without their '
            'knowledge; one group being supplied with water containing the sewage of London, and, '
            'amongst it, whatever might have come from the cholera patients, the other group having '
            'water quite free from such impurity."'
        ),
        "question": (
            "Snow claims this natural situation is equivalent to a planned experiment. What specific "
            "features of this situation does Snow say eliminate confounding? Why does Snow emphasize "
            "that the division into two groups occurred 'without their choice, and, in most cases, "
            "without their knowledge'?"
        ),
        "correct_answer": (
            "Features eliminating confounding: same houses, same people, same physical conditions — "
            "everything identical except water supply. Snow emphasizes lack of choice and knowledge "
            "to rule out self-selection bias: people could not choose the safer water, so any "
            "difference in outcomes must be due to the water supply, not to the behaviour or "
            "preferences of the people. This is functionally equivalent to random assignment."
        ),
        "score_keywords": [
            ["no difference", "same houses", "same people", "identical", "physical conditions"],
            ["choice", "knowledge", "self-selection", "random", "did not choose"],
            ["confound", "bias", "water supply", "only difference", "equivalent"],
        ],
    },
    {
        "id": 4,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd edition, 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"As soon as I became acquainted with the situation and extent of this irruption of cholera, '
            'I suspected some contamination of the water of the much-frequented street pump in Broad '
            'Street, near the end of Cambridge Street; but on examining the water, on the evening of '
            'the 3rd September, I found so little impurity in it of an organic nature, that I hesitated '
            'to come to a conclusion. Further inquiry, however, showed me that there was no other '
            'circumstance or agent common to the circumscribed locality in which this sudden increase '
            'of cholera occurred, and not extending beyond it, except the water of the above mentioned pump."'
        ),
        "question": (
            "Snow says he 'hesitated to come to a conclusion' after examining the water. What evidence "
            "did he use instead to overcome that hesitation? What logical method is Snow applying when "
            "he says there was 'no other circumstance or agent common' to the affected area except the "
            "pump water?"
        ),
        "correct_answer": (
            "Snow overcame his hesitation through elimination — ruling out every other possible common "
            "factor in the affected locality. The logical method is Mill's Method of Agreement (and "
            "elimination): the pump water was the only circumstance common to all cases and absent "
            "in unaffected areas. The chemical test was insufficient — the epidemiological pattern "
            "of elimination was stronger evidence than the direct physical examination."
        ),
        "score_keywords": [
            ["eliminat", "ruled out", "no other", "only factor", "common"],
            ["Mill", "method of agreement", "logical", "induction", "inference"],
            ["epidemiolog", "pattern", "stronger", "chemical test", "insufficient", "physical examination"],
        ],
    },
    {
        "id": 5,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Second Canon and example, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"If an instance in which the phenomenon under investigation occurs, and an instance in which '
            'it does not occur, have every circumstance in common save one, that one occurring only in '
            'the former; the circumstance in which alone the two instances differ, is the effect, or '
            'the cause, or an indispensable part of the cause, of the phenomenon."\n\n'
            'And from Mill\'s example immediately following:\n\n'
            '"It is scarcely necessary to give examples of a logical process to which we owe almost all '
            'the inductive conclusions we draw in daily life. When a man is shot through the heart, it '
            'is by this method we know that it was the gunshot which killed him: for he was in the '
            'fullness of life immediately before, all circumstances being the same, except the wound."'
        ),
        "question": (
            "In the gunshot example, identify: (1) the two instances being compared, (2) what they have "
            "in common, (3) the one circumstance in which they differ, and (4) what Mill's method "
            "concludes from this. Why does Mill say this method is responsible for 'almost all the "
            "inductive conclusions we draw in daily life'?"
        ),
        "correct_answer": (
            "(1) Two instances: the man immediately before the shot (alive) and the man after the shot "
            "(dead). (2) In common: all circumstances — physical state, environment, everything. "
            "(3) The one difference: the gunshot wound. (4) Mill's conclusion: the wound is the cause "
            "of death. Mill says this method underlies everyday causal inference because we constantly "
            "compare a situation before and after a single change to identify what caused a difference."
        ),
        "score_keywords": [
            ["before", "after", "two instances", "alive", "fullness of life"],
            ["all circumstances", "everything the same", "every circumstance"],
            ["wound", "gunshot", "one difference", "differs", "save one"],
        ],
    },
    {
        "id": 6,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"In comparing the deaths of one hospital with those of another, any statistics are justly '
            'considered absolutely valueless which do not give the ages, the sexes, and the diseases '
            'of all the cases. It does not seem necessary to mention this. It does not seem necessary '
            'to say that there can be no comparison between old men with dropsies and young women with '
            'consumptions. Yet the cleverest men and the cleverest women are often heard making such '
            'comparisons, ignoring entirely sex, age, disease, place--in fact, all the conditions '
            'essential to the question. It is the merest gossip."'
        ),
        "question": (
            "Nightingale calls unadjusted hospital death-rate comparisons 'the merest gossip' and "
            "'absolutely valueless.' What is the statistical problem she is identifying? What variables "
            "does she say must be accounted for, and why would ignoring them lead to a wrong conclusion "
            "about which hospital is better?"
        ),
        "correct_answer": (
            "Nightingale is identifying confounding — hospitals treat different types of patients, so "
            "a higher death rate may reflect a sicker patient population, not worse care. Variables "
            "required: age, sex, disease type, and place. Without adjusting for these, the comparison "
            "measures patient mix rather than hospital quality. Comparing 'old men with dropsies' to "
            "'young women with consumptions' is invalid because their baseline mortality is completely "
            "different."
        ),
        "score_keywords": [
            ["confound", "patient mix", "different patients", "sicker", "case mix"],
            ["age", "sex", "disease", "adjust", "account for"],
            ["invalid", "wrong", "mislead", "baseline", "mortality", "patient population"],
        ],
    },
]


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_answer(answer: str, keyword_groups: list) -> tuple[int, int]:
    """Score answer by checking if at least one keyword from each group appears."""
    answer_lower = answer.lower()
    hits = 0
    for group in keyword_groups:
        if any(kw.lower() in answer_lower for kw in group):
            hits += 1
    return hits, len(keyword_groups)


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_models(quantize: bool):
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    quant_cfg = None
    if quantize:
        print("Quantization: 4-bit (bitsandbytes)")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print("Loading base model (this takes ~90s)...")
    dtype = torch.float16 if not quantize else None
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading TunedAI Labs adapter...")
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    tuned_model.eval()

    print("✓ Models ready.\n")
    return tokenizer, tuned_model


def ask(question: str, tokenizer, model, use_adapter: bool) -> str:
    if use_adapter:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.1,
            do_sample=False,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true",
                        help="Use 4-bit quantization (for limited VRAM)")
    parser.add_argument("--base-only", action="store_true",
                        help="Only run base model (faster, skips adapter)")
    args = parser.parse_args()

    tokenizer, model = load_models(args.quantize)

    results = []
    base_total_hits = base_total_pts = 0
    tuned_total_hits = tuned_total_pts = 0

    SEP  = "=" * 72
    DIV  = "-" * 72

    for q in QUESTIONS:
        prompt = f"VERBATIM PASSAGE ({q['source'].split(',')[0]}):\n\n{q['passage']}\n\nQUESTION: {q['question']}"

        print(SEP)
        print(f"TEST {q['id']} — {q['source']}")
        print(f"Retrieved from: {q['retrieved_from']}")
        print(SEP)
        print(f"PASSAGE:\n{q['passage']}\n")
        print(f"QUESTION:\n{q['question']}\n")

        # Base model
        print(DIV)
        print("[ BASE QWEN 2.5-7B — untuned ]")
        print(DIV)
        base_ans = ask(prompt, tokenizer, model, use_adapter=False)
        print(base_ans)
        base_hits, base_pts = score_answer(base_ans, q["score_keywords"])
        print(f"\n  → Score: {base_hits}/{base_pts} keyword groups matched")

        result = {
            "id": q["id"],
            "source": q["source"],
            "retrieved_from": q["retrieved_from"],
            "correct_answer": q["correct_answer"],
            "base": {"answer": base_ans, "hits": base_hits, "points": base_pts},
        }

        base_total_hits += base_hits
        base_total_pts  += base_pts

        # Tuned model
        if not args.base_only:
            print(DIV)
            print("[ TUNEDAI LABS — fine-tuned for reasoning ]")
            print(DIV)
            tuned_ans = ask(prompt, tokenizer, model, use_adapter=True)
            print(tuned_ans)
            tuned_hits, tuned_pts = score_answer(tuned_ans, q["score_keywords"])
            print(f"\n  → Score: {tuned_hits}/{tuned_pts} keyword groups matched")

            result["tuned"] = {"answer": tuned_ans, "hits": tuned_hits, "points": tuned_pts}
            tuned_total_hits += tuned_hits
            tuned_total_pts  += tuned_pts

        print(f"\n  CORRECT ANSWER:\n  {q['correct_answer']}\n")
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(SEP)
    print("FINAL SCORES — Raw Passage Reasoning Test")
    print(f"6 verbatim passages · Hume (1748), Snow (1855), Mill (1843), Nightingale (1860)")
    print(SEP)
    base_pct = round(100 * base_total_hits / base_total_pts, 1)
    print(f"Base Qwen 2.5-7B (untuned):       {base_total_hits}/{base_total_pts} = {base_pct}%")

    if not args.base_only:
        tuned_pct = round(100 * tuned_total_hits / tuned_total_pts, 1)
        delta = round(tuned_pct - base_pct, 1)
        print(f"TunedAI Labs (reasoning-tuned):    {tuned_total_hits}/{tuned_total_pts} = {tuned_pct}%")
        print(f"Delta:                             +{delta}%")

    print(SEP)
    print("NOTE: Keyword scoring is a rough indicator. Read full answers above for qualitative assessment.")
    print(SEP)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "test": "TunedAI Labs Raw Passage Reasoning Test",
        "run_at": datetime.datetime.utcnow().isoformat() + "Z",
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_REPO,
        "quantized": args.quantize,
        "summary": {
            "base": {"hits": base_total_hits, "points": base_total_pts, "pct": base_pct},
        },
        "questions": results,
    }
    if not args.base_only:
        output["summary"]["tuned"] = {
            "hits": tuned_total_hits, "points": tuned_total_pts, "pct": tuned_pct
        }
        output["summary"]["delta_pct"] = delta

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
