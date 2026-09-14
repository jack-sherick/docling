"""
Eval harness: run the RFP-comparison test case across chunking strategies x
LLMs, and log token usage for each scenario.

For each (chunking strategy, model) pair:
  1. Retrieve relevant chunks per selected balancing authority (BA) from that
     strategy's Chroma collection (built by embed_chunks.py).
  2. Send one prompt to the model asking it to compare the LSE solicitations
     and produce the values table + narrative.
  3. Record input/output token usage and save the raw response.

Vector DB is fixed to Chroma (local, already built) for every scenario, so
chunking strategy and model are the two variables under test.

Requires:
  - Chroma collections already built: `python embed_chunks.py --strategy small`
    and `python embed_chunks.py --strategy large`
  - OPENAI_API_KEY and ANTHROPIC_API_KEY set as environment variables
  - Ollama running locally with the llama3.1:8b model pulled

Usage:
    python run_eval.py
    python run_eval.py --bas Avista "Idaho Power" "Puget Sound Energy"
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from embed_chunks import STRATEGIES, DB_DIR, EMBEDDING_MODEL

RESULTS_DIR = Path(__file__).parent / "eval-results"

# Maps a human-facing BA name (what a dropdown would show) to the Docling
# source_file stem(s) that make up that LSE's solicitation. Avista's RFP is
# split across a main document and a separate requirements exhibit.
BA_TO_FILES = {
    "Avista": [
        "Avista 2025 All-Source RFP.pdf",
        "Avista Exhibit C - Detailed Proposal Requirements.pdf",
    ],
    "Idaho Power": ["Idaho 2032_IPC_AllSource_RFP_final vfinal.pdf"],
    "PacifiCorp West": ["Pacificorp 2025_WA_Situs_RFP_Main_Document.pdf"],
    "Puget Sound Energy": ["Puget Sound Energy 2026 Voluntary Utility-Scale RFP_013026.pdf"],
    "BC Hydro": ["BC Hydro 2025-cfp-request-for-proposals.pdf"],
    "SJCE (AVA)": ["AVA 9-19-24-Long-Term-Resource-RFO-Solicitation-Protocol.pdf"],
    "CC-Power": ["CC-Power-2025-All-Source-RFP-Instructions-FINAL-1.pdf"],
    "Portland General": ["Portland General 2025_All-Source_RFP_Main_Document.pdf"],
    "PWP": [
        "PWP Design-Build_Services_for_Solar_and_Energy_Storage_Installations_at_Category_1_City-Owned_Properties.pdf"
    ],
    "SDCP": ["SDCP-PV-Term-Sheet-Template-2025_DAC-GT_FINAL.pdf"],
    "SDG&E": ["SDGE 2028-2031 Firm Zero Emitting IRP Reliability RFO Protocol Reopening.pdf"],
}

# One retrieval query per fact the prompt asks for, so multi-fact answers
# aren't dependent on a single similarity search finding everything at once.
QUERIES = [
    "identified need or driver for this energy resource solicitation",
    "total MW or GWh capacity requested in this solicitation",
    "type of resource being solicited and minimum or mandated resource requirements",
    "expected or preferred commercial operation date COD",
    "interconnection and deliverability requirements for resources",
]

PROMPT_TEMPLATE = """You are analyzing utility RFP/RFO solicitations for load-serving entities (LSEs). Below are retrieved excerpts from each LSE's solicitation document(s).

{context}

---

Compare the RFPs and identify the following values for each LSE solicitation:
- The identified drivers for the energy resource solicitation
- Total MW or GWh Capacity requested in the solicitation
- The type of resources being solicited and any minimum or mandated requirements for a particular resource
- The expected or preferred Commercial Operation Date
- The interconnection and deliverability requirements for the resources

Produce a table with the LSE values. In 2-3 paragraph narrative, summarize the energy resource requirement in the area and any significant clustering of resource types or interconnection points that may impact the overall ability of the area to obtain the necessary resources for the area.

Base your answer only on the excerpts provided above. If a value isn't present in the excerpts, say "Not found in retrieved context" rather than guessing.
"""

MODELS = [
    {
        "key": "local-llama3.1-8b",
        "provider": "ollama",
        "model": "llama3.1:8b",
        "company": "Meta (via Ollama, local)",
        "version": "Llama 3.1 8B (released 2024-07-23)",
    },
    {
        "key": "anthropic-sonnet5",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "company": "Anthropic",
        "version": "claude-sonnet-5 (via Claude Code CLI / Pro subscription, not metered API)",
    },
    {
        "key": "openai-gpt5.4",
        "provider": "openai",
        "model": "gpt-5.4-2026-03-05",
        "company": "OpenAI",
        "version": "gpt-5.4, 2026-03-05 snapshot",
    },
]


def retrieve_context(collection, embed_model, ba_name: str, source_files: list[str], top_k: int = 3) -> str:
    seen = {}
    for query in QUERIES:
        query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"source_file": {"$in": source_files}},
        )
        for chunk_id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
            seen[chunk_id] = (doc, meta)

    ordered = sorted(seen.items(), key=lambda kv: (kv[1][1]["source_file"], int(kv[0].rsplit("::", 1)[-1])))
    lines = [f"=== {ba_name} ==="]
    for _, (doc, meta) in ordered:
        lines.append(f"[{meta['source_file']} p.{meta['page_numbers']}]\n{doc}")
    return "\n\n".join(lines)


def call_ollama(prompt: str, model: str) -> tuple[str, int | None, int | None]:
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Ollama defaults to a 2048-token context window regardless of what
            # the model architecture supports, silently truncating anything
            # longer. Our contexts run well past that, so raise it explicitly.
            "options": {"num_ctx": 32768},
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"], data.get("prompt_eval_count"), data.get("eval_count")


def call_openai(prompt: str, model: str) -> tuple[str, int | None, int | None]:
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def call_anthropic(prompt: str, model: str) -> tuple[str, int | None, int | None]:
    """Routes through the local Claude Code CLI (authenticated to this
    machine's Claude.ai subscription) instead of the pay-per-token Anthropic
    API, so this draws from Pro/Max plan usage rather than a separate bill.
    Flags strip Claude Code's own tool/skill/system-prompt scaffolding down to
    a plain completion call; prompt goes over stdin to dodge Windows' ~32K
    command-line argument limit on the larger contexts."""
    result = subprocess.run(
        [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            model,
            "--system-prompt",
            "You are a helpful assistant.",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--disallowedTools",
            "*",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    result.check_returncode()
    data = json.loads(result.stdout)
    usage = data["usage"]
    input_tokens = (
        usage["input_tokens"] + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    )
    return data["result"], input_tokens, usage["output_tokens"]


CALLERS = {"ollama": call_ollama, "openai": call_openai, "anthropic": call_anthropic}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bas",
        nargs="+",
        default=["Avista", "Idaho Power", "Puget Sound Energy"],
        help=f"Balancing authorities to include. Choices: {list(BA_TO_FILES)}",
    )
    args = parser.parse_args()

    unknown = [ba for ba in args.bas if ba not in BA_TO_FILES]
    if unknown:
        print(f"Unknown BA(s): {unknown}. Choices: {list(BA_TO_FILES)}")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    rows = []
    for strategy_name, (_, collection_name) in STRATEGIES.items():
        collection = client.get_or_create_collection(collection_name)
        context = "\n\n".join(
            retrieve_context(collection, embed_model, ba, BA_TO_FILES[ba]) for ba in args.bas
        )
        prompt = PROMPT_TEMPLATE.format(context=context)
        prompt_path = RESULTS_DIR / f"{strategy_name}__prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"\n[{strategy_name}] context built ({len(context)} chars) -> {prompt_path}")

        for model_cfg in MODELS:
            print(f"  Running model={model_cfg['key']} ...")
            try:
                text, in_tok, out_tok = CALLERS[model_cfg["provider"]](prompt, model_cfg["model"])
            except Exception as e:
                print(f"  FAILED ({model_cfg['key']}): {e}")
                rows.append(
                    {
                        "chunking_strategy": strategy_name,
                        "vector_db": "Chroma",
                        "company": model_cfg["company"],
                        "model": model_cfg["model"],
                        "version": model_cfg["version"],
                        "input_tokens": "ERROR",
                        "output_tokens": "ERROR",
                        "total_tokens": "ERROR",
                        "output_file": str(e),
                    }
                )
                continue

            out_path = RESULTS_DIR / f"{strategy_name}__{model_cfg['key']}.md"
            out_path.write_text(text, encoding="utf-8")
            total = (in_tok or 0) + (out_tok or 0)
            rows.append(
                {
                    "chunking_strategy": strategy_name,
                    "vector_db": "Chroma",
                    "company": model_cfg["company"],
                    "model": model_cfg["model"],
                    "version": model_cfg["version"],
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": total,
                    "output_file": str(out_path),
                }
            )
            print(f"    input={in_tok} output={out_tok} total={total} -> {out_path}")

    summary_path = RESULTS_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Strategy':<10} {'Company':<25} {'Model':<25} {'Input':>8} {'Output':>8} {'Total':>8}")
    for r in rows:
        print(
            f"{r['chunking_strategy']:<10} {r['company']:<25} {r['model']:<25} "
            f"{str(r['input_tokens']):>8} {str(r['output_tokens']):>8} {str(r['total_tokens']):>8}"
        )
    print(f"\nSummary CSV -> {summary_path}")
    print(f"Per-scenario responses -> {RESULTS_DIR}\\<strategy>__<model_key>.md")


if __name__ == "__main__":
    main()
