"""
Complex and long prompt compression test.
Runs multi-paragraph prompts, complex system instructions, and multi-constraint queries
through the optimizer and saves the results to CSV.

Usage:
    uv run python test_complex.py
"""
from __future__ import annotations

import csv
import sys
import io
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
from prompt_optimizer.training.config import TrainingConfig

# ---------------------------------------------------------------------------
# Complex, multi-paragraph, and multi-constraint prompts (100 - 400+ words)
# ---------------------------------------------------------------------------
COMPLEX_PROMPTS = [
    (
        # 1. System Prompt for structured JSON agent
        "You are a highly structured data extraction assistant. Your task is to extract information from "
        "the provided unstructured customer emails and format the output as a strict JSON object. "
        "You must adhere to the following schema:\n"
        "{\n"
        "  \"customer_name\": \"string or null\",\n"
        "  \"urgency\": \"low, medium, high\",\n"
        "  \"category\": \"billing, technical_support, feedback, other\",\n"
        "  \"summary\": \"a brief 1-sentence summary of the request\"\n"
        "}\n"
        "Do not include any conversational filler, markdown formatting (like ```json), or explanatory text. "
        "If you cannot find the customer's name, set it to null. Answer only in JSON.",
        "general"
    ),
    (
        # 2. Multi-constraint Python programming request
        "Please write a high-performance Python function that reads a large log file (approx 10GB) line by line, "
        "extracts all IP addresses, and counts their frequencies. You must satisfy the following constraints:\n"
        "1. Do not load the entire file into memory at once to prevent Out of Memory (OOM) errors.\n"
        "2. Use regular expressions to validate and extract the IP addresses.\n"
        "3. Use a multithreading or multiprocessing approach to speed up processing if possible, but keep thread safety in mind.\n"
        "4. Return the top 10 most frequent IP addresses as a list of tuples containing the IP and its count.\n"
        "5. Write unit tests for this function using the unittest framework.",
        "code"
    ),
    (
        # 3. SQL Query and schema design
        "You are given a database schema with three tables: users (id, name, created_at), "
        "orders (id, user_id, total_amount, status, order_date), and order_items (id, order_id, product_id, quantity, price). "
        "Please write an optimized SQL query that retrieves the name of all users who have spent more than $1000 "
        "in total across all their orders, but only for orders that were placed in the year 2025. "
        "In addition, order the results by the total amount spent in descending order, and explain any indexes "
        "that should be created on these tables to ensure the query runs efficiently on a database with millions of rows.",
        "code"
    ),
    (
        # 4. Complex Business Strategy Case Analysis
        "Act as a senior business consultant. Analyze the following scenario and provide a comprehensive SWOT "
        "analysis along with 3 strategic recommendations. A mid-sized retail company specializing in organic grocery "
        "products has been operating brick-and-mortar stores in suburban areas for 15 years. They are experiencing "
        "declining foot traffic due to aggressive expansion by large supermarket chains and rapid adoption of "
        "online grocery delivery services. Their strengths include a loyal customer base and strong local farm partnerships. "
        "However, they lack an e-commerce platform and their operating margins are shrinking. Focus on how they can leverage "
        "their local partnerships to differentiate themselves online.",
        "general"
    ),
    (
        # 5. Multi-part Academic Summarization
        "Read the following description of the CRISPR-Cas9 gene-editing system and summarize its mechanism and "
        "ethical implications in exactly two paragraphs. CRISPR-Cas9 is a unique technology that enables geneticists "
        "and medical researchers to edit parts of the genome by removing, adding, or altering sections of the DNA sequence. "
        "It is currently the simplest, most versatile, and precise method of genetic manipulation. The CRISPR-Cas9 system "
        "consists of two key molecules that introduce a change into the DNA: an enzyme called Cas9, which acts as a pair of "
        "molecular scissors capable of cutting DNA strands, and a piece of RNA called guide RNA (gRNA), which directs the Cas9 "
        "enzyme to the specific genomic location. While it holds promise for treating genetic diseases, it raises significant "
        "ethical concerns regarding off-target mutations, germline editing, and ecological impacts of gene drives.",
        "summarization"
    ),
    (
        # 6. Detailed API Integration / OAuth request
        "I need to integrate the Stripe payment gateway into my Node.js web application. Please provide a complete, "
        "step-by-step tutorial and working code example that handles a subscription checkout flow. Specifically, I need "
        "you to explain how to create a Checkout Session on the backend, redirect the user to Stripe's hosted payment page, "
        "and handle the webhook event (invoice.payment_succeeded) securely to provision the subscription in my PostgreSQL database. "
        "Make sure to explain how to verify the Stripe webhook signature to prevent spoofing attacks, and handle errors gracefully.",
        "code"
    ),
    (
        # 7. Algorithm Design & Complex Constraints (Trie)
        "Write a clean, documented Python implementation of a Trie (Prefix Tree) data structure that supports the "
        "insert, search, and startsWith operations. Once you have implemented the basic Trie, add a custom method called "
        "find_words_with_prefix that returns all words in the Trie that start with a given prefix, sorted by alphabetical order. "
        "Explain the time and space complexity of each operation (insert, search, prefix search) in Big O notation, and "
        "explain how this structure is more efficient than using a standard hash map or list for prefix matching tasks.",
        "code"
    ),
    (
        # 8. Detailed Roleplay and Tone Constraint
        "You are an AI assistant representing a luxury hotel chain. A customer is writing an email complaining that "
        "their room was not ready upon arrival, their luggage was delayed by 3 hours, and the staff at the front desk "
        "was impolite. Write a professional, empathetic, and sophisticated email response. You must apologize sincerely, "
        "explain that these incidents do not reflect our standard of service, offer them a complimentary room upgrade "
        "for their next stay along with a $200 dining credit, and request their contact details to follow up personally. "
        "Maintain a highly polished, warm, and elite tone throughout the response.",
        "creative"
    )
]

def main() -> None:
    ckpt_path = "checkpoints/sweep_lambda_0_1/final.pt"
    if not Path(ckpt_path).exists():
        ckpt_path = "checkpoints/final.pt"

    out_path = Path("results/complex_prompts_test.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n  Loading model from {ckpt_path} ...")
    cfg = TrainingConfig()
    model = PromptOptimizerModel(cfg)
    model.load(ckpt_path)
    model.eval()
    tokenizer = model.token_scorer.tokenizer

    print(f"  Running {len(COMPLEX_PROMPTS)} complex, multi-paragraph prompts ...\n")

    rows = []
    for idx, (prompt, task) in enumerate(COMPLEX_PROMPTS, 1):
        start = time.perf_counter()
        compressed = model.compress([prompt], [task])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        orig_tokens = len(tokenizer(prompt)["input_ids"])
        comp_tokens = len(tokenizer(compressed)["input_ids"])
        orig_words = len(prompt.split())
        comp_words = len(compressed.split())
        token_reduction = (1 - comp_tokens / max(orig_tokens, 1)) * 100
        word_reduction = (1 - comp_words / max(orig_words, 1)) * 100

        rows.append({
            "id": idx,
            "task_type": task,
            "original_prompt": prompt,
            "compressed_prompt": compressed,
            "original_words": orig_words,
            "compressed_words": comp_words,
            "word_reduction_pct": round(word_reduction, 1),
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "token_reduction_pct": round(token_reduction, 1),
            "compression_time_ms": round(elapsed_ms, 1),
        })

        # Print detailed report for each prompt
        print(f"--- Prompt #{idx} ({task.upper()}) ---")
        print(f"Original Words: {orig_words} | Tokens: {orig_tokens}")
        print(f"Compressed Words: {comp_words} | Tokens: {comp_tokens} ({token_reduction:.1f}% reduction)")
        print(f"Time: {elapsed_ms:.1f}ms")
        print("\n[Compressed Output]:")
        print(compressed)
        print("=" * 80 + "\n")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved complex prompts batch test results to: {out_path}\n")

if __name__ == "__main__":
    main()
