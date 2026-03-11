
import os
import re
from pathlib import Path
from openai import OpenAI

INFERENCE_FILE = Path("inference_results.md")
OUTPUT_FILE = Path("chatgpt_articles.md")
MODEL = "gpt-4o-mini"

def parse_inference_results(path: Path) -> list[dict]:
    """Return a list of {question, yes, no} dicts from the markdown file."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"-\s+\*\*Question:\*\*\s+(.+?)\n\s+YES:\s+([\d.]+)\n\s+NO:\s+([\d.]+)",
        re.MULTILINE,
    )

    results = []
    for match in pattern.finditer(text):
        results.append(
            {
                "question": match.group(1).strip(),
                "yes": float(match.group(2)),
                "no": float(match.group(3)),
            }
        )
    return results


def build_prompt(result: dict) -> str:
    yes_pct = result["yes"] * 100
    no_pct = result["no"] * 100
    if result["yes"] >= 0.75:
        confidence_desc = f"highly likely (YES: {yes_pct:.1f}%)"
    elif result["yes"] >= 0.55:
        confidence_desc = f"somewhat likely (YES: {yes_pct:.1f}%)"
    elif result["no"] >= 0.75:
        confidence_desc = f"highly unlikely (NO: {no_pct:.1f}%)"
    else:
        confidence_desc = f"uncertain (YES: {yes_pct:.1f}% / NO: {no_pct:.1f}%)"

    return (
        f"A prediction model evaluated the following yes/no question:\n\n"
        f'"{result["question"]}"\n\n'
        f"The model considers this outcome {confidence_desc}.\n\n"
        f"Write a concise, informative article of approximately 200 words about this topic. "
        f"Incorporate the prediction confidence into your analysis — discuss why the outcome "
        f"might or might not happen, and what factors drive the uncertainty. "
        f"Write in a neutral, journalistic tone."
    )

def generate_article(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable journalist who writes clear, balanced short articles "
                    "about predictions and future events. Keep articles around 200 words."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Set it with:  $env:OPENAI_API_KEY = 'sk-...'"
        )

    client = OpenAI(api_key=api_key)

    results = parse_inference_results(INFERENCE_FILE)
    if not results:
        print(f"No predictions found in {INFERENCE_FILE}.")
        return

    print(f"Found {len(results)} prediction(s). Generating articles...\n")

    output_lines = ["# ChatGPT Articles from NBOW Predictions", ""]

    for i, result in enumerate(results, 1):
        print(f"[{i}/{len(results)}] {result['question']}")
        prompt = build_prompt(result)
        article = generate_article(client, prompt)

        output_lines += [
            f"## {i}. {result['question']}",
            "",
            f"**Model confidence** — YES: {result['yes']*100:.1f}%  |  NO: {result['no']*100:.1f}%",
            "",
            article,
            "",
        ]
        print(f"  -> Done ({len(article.split())} words)\n")

    OUTPUT_FILE.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Saved all articles to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
