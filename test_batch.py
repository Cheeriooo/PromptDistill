"""
Batch compression test — runs diverse prompts through the optimizer
and saves results to CSV for manual review.

Usage:
    uv run python test_batch.py
    uv run python test_batch.py --lambda 0.05
    uv run python test_batch.py --output results/my_test.csv
"""
from __future__ import annotations

import argparse
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
# Test prompts — 100 diverse prompts across all task types
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    # ======================== QA — General Knowledge ========================
    ("What is the capital of india", "qa"),
    ("What is the capital of France", "qa"),
    ("Why is the sky blue", "qa"),
    ("Who invented the telephone", "qa"),
    ("How does machine learning work", "qa"),
    ("When was the Eiffel Tower built", "qa"),
    ("Where is the Amazon Rainforest located", "qa"),
    ("What is the speed of light in a vacuum", "qa"),
    ("What are the 3 laws of thermodynamics", "qa"),
    ("What is DNA and how does it store genetic information", "qa"),
    ("How does the human immune system fight off infections", "qa"),
    ("What causes earthquakes and how are they measured", "qa"),
    ("What is the difference between a virus and a bacteria", "qa"),
    ("How do black holes form and what happens inside them", "qa"),
    ("Why do we dream and what purpose do dreams serve", "qa"),

    # ======================== QA — Comparisons ========================
    ("What is the difference between Python and JavaScript", "qa"),
    ("What is the difference between a compiler and an interpreter", "qa"),
    ("Explain the pros and cons of microservices versus monolith architecture", "qa"),
    ("What are the advantages and disadvantages of using NoSQL databases compared to SQL databases", "qa"),
    ("Compare the iPhone 15 Pro Max with Samsung Galaxy S24 Ultra", "qa"),
    ("What is the difference between a stack and a queue in data structures", "qa"),
    ("Compare supervised learning and unsupervised learning in machine learning", "qa"),
    ("What is the difference between HTTP and HTTPS", "qa"),
    ("Compare TCP and UDP protocols and when to use each", "qa"),
    ("What is the difference between a process and a thread in operating systems", "qa"),

    # ======================== QA — History & Culture ========================
    ("Could you please provide me with a comprehensive and detailed summary of the key events that led to the First World War?", "qa"),
    ("How did the Renaissance influence modern European art and culture", "qa"),
    ("Explain the significance of the Magna Carta in British history", "qa"),
    ("What were the main causes and consequences of the French Revolution", "qa"),
    ("How did the Industrial Revolution change society and the economy", "qa"),
    ("What is the history of the United Nations and its role in global peace", "qa"),
    ("Summarize the plot of Harry Potter and the Chamber of Secrets", "qa"),
    ("What did Albert Einstein contribute to physics", "qa"),

    # ======================== QA — Science & Math ========================
    ("What is the time complexity of quicksort in the worst case", "qa"),
    ("Calculate the area of a circle with radius 5", "qa"),
    ("What is the boiling point of water at 1 atmosphere of pressure", "qa"),
    ("How long does it take light to travel 1 light year", "qa"),
    ("Explain quantum computing in simple terms for a beginner", "qa"),
    ("What is the Heisenberg uncertainty principle and why does it matter", "qa"),
    ("How does the theory of general relativity explain gravity", "qa"),
    ("What is Euler's identity and why is it considered beautiful in mathematics", "qa"),

    # ======================== Code — Algorithms ========================
    ("Can you please provide me code for binary search", "code"),
    ("Write a Python function to merge two sorted linked lists", "code"),
    ("How do I implement a binary tree traversal in Java", "code"),
    ("Implement a thread-safe singleton pattern in Java using double-checked locking", "code"),
    ("Write a recursive function in C++ to calculate the Fibonacci sequence", "code"),
    ("Can you please help me write a detailed Python function that reads a CSV file, processes the data by removing duplicates and null values?", "code"),
    ("Write a function in Python that checks if a string is a valid palindrome ignoring spaces and punctuation", "code"),
    ("Implement a LRU cache in Python with O(1) get and put operations", "code"),
    ("Write a Python function to find the longest common subsequence of two strings", "code"),
    ("Implement depth-first search and breadth-first search for a graph in Python", "code"),

    # ======================== Code — Debugging & Constraints ========================
    ("Debug this SQL query that returns duplicate rows", "code"),
    ("Could you help me debug this Python error", "code"),
    ("Write Python code that does not use recursion for factorial", "code"),
    ("Don't use any global variables in the code", "code"),
    ("Write a Python function that sorts a list in descending order without using the built-in sort method", "code"),
    ("I can't understand why this function returns None", "code"),
    ("Write a REST API endpoint in Flask that handles pagination with limit and offset parameters", "code"),
    ("Create a Python decorator that retries a function up to 3 times on failure", "code"),

    # ======================== Code — Specific Technologies ========================
    ("Explain how OAuth 2.0 authentication works", "code"),
    ("Create a JSON schema for a user profile with name, email, and age fields", "code"),
    ("Write a Docker Compose file for a Python Flask app with PostgreSQL and Redis", "code"),
    ("How do I set up a CI/CD pipeline using GitHub Actions for a Node.js project", "code"),
    ("Write a Terraform configuration to deploy an AWS Lambda function", "code"),
    ("Explain how WebSockets differ from HTTP polling and when to use each", "code"),

    # ======================== Creative Writing ========================
    ("Write me a poem about love", "creative"),
    ("Create a short horror story set in an abandoned hospital", "creative"),
    ("Write a professional email to decline a job offer politely", "creative"),
    ("Write a haiku about artificial intelligence", "creative"),
    ("Create a motivational speech for high school graduates about pursuing their dreams", "creative"),
    ("Write a product description for a smart water bottle that tracks hydration", "creative"),
    ("Compose a limerick about a programmer who loves coffee", "creative"),
    ("Write a bedtime story for children about a friendly dragon who learns to share", "creative"),

    # ======================== Summarization ========================
    ("Please summarize this article for me in a few sentences", "summarization"),
    ("Give me a brief overview of the main points in this research paper", "summarization"),
    ("Condense this meeting transcript into action items and key decisions", "summarization"),
    ("Summarize the key takeaways from this quarterly financial report", "summarization"),

    # ======================== General — Verbose Prompts ========================
    ("I would really appreciate it if you could explain the concept of blockchain technology and how it is used in cryptocurrency", "general"),
    ("Would you be so kind as to provide me with a detailed explanation of how neural networks learn through the process of backpropagation", "general"),
    ("I was wondering if you could possibly help me understand the process of photosynthesis and explain it in a way that would be easy for a high school student to comprehend", "general"),
    ("Hey, I was just wondering if you could maybe help me out with something. I need to write a REST API in Python using Flask that handles CRUD operations for a todo list application", "general"),
    ("So basically what I need is for you to explain to me in very simple terms that even a child could understand how exactly does the internet work and what happens when you type a URL in the browser", "general"),

    # ======================== General — Short & Direct ========================
    ("Translate this to French", "general"),
    ("Sort this list", "general"),
    ("Explain recursion", "general"),
    ("Define entropy", "general"),
    ("What is AI", "general"),
    ("Fix this bug", "general"),
    ("Hello world", "general"),
    ("Define love", "general"),
    ("Explain polymorphism", "general"),
    ("Convert Celsius to Fahrenheit", "general"),

    # ======================== Edge Cases ========================
    ("List the top 5 programming languages in 2024", "qa"),
    ("Give me 10 creative names for a coffee shop", "creative"),
    ("Compare React, Angular, and Vue for frontend development", "code"),
    ("Compare the GDP of United States, China, and Germany", "qa"),
    ("List the planets in our solar system in order from the Sun", "qa"),
    ("Explain the difference between TCP and UDP without technical jargon", "qa"),
    ("List 5 countries that are not in Europe", "qa"),
    ("It is not uncommon for neural networks to not converge", "qa"),
    ("My Python script is slow. How can I optimize it for performance?", "code"),
    ("I have a list of numbers. Write code to find the median.", "code"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test prompt optimizer and export to CSV")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=0.1,
                        help="Compression lambda (default: 0.1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (auto-resolved if not given)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: results/batch_test_lambda_X.csv)")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        sweep_paths = {
            0.01: "checkpoints/sweep_lambda_0_01/final.pt",
            0.05: "checkpoints/sweep_lambda_0_05/final.pt",
            0.1: "checkpoints/sweep_lambda_0_1/final.pt",
            0.5: "checkpoints/sweep_lambda_0_5/final.pt",
        }
        ckpt_path = sweep_paths.get(args.lambda_val, "checkpoints/final.pt")
        if not Path(ckpt_path).exists():
            ckpt_path = "checkpoints/final.pt"

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        lambda_str = str(args.lambda_val).replace(".", "_")
        out_path = Path(f"results/batch_test_lambda_{lambda_str}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n  Loading model from {ckpt_path} ...")
    cfg = TrainingConfig()
    model = PromptOptimizerModel(cfg)
    model.load(ckpt_path)
    model.eval()
    tokenizer = model.token_scorer.tokenizer

    print(f"  Running {len(TEST_PROMPTS)} prompts (lambda={args.lambda_val}) ...\n")

    # Run all prompts
    rows = []
    total_orig_tokens = 0
    total_comp_tokens = 0

    for idx, (prompt, task) in enumerate(TEST_PROMPTS, 1):
        start = time.perf_counter()
        compressed = model.compress([prompt], [task])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        orig_tokens = len(tokenizer(prompt)["input_ids"])
        comp_tokens = len(tokenizer(compressed)["input_ids"])
        orig_words = len(prompt.split())
        comp_words = len(compressed.split())
        token_reduction = (1 - comp_tokens / max(orig_tokens, 1)) * 100
        word_reduction = (1 - comp_words / max(orig_words, 1)) * 100

        total_orig_tokens += orig_tokens
        total_comp_tokens += comp_tokens

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
            "quality_ok": "",  # For manual review — leave blank
            "notes": "",       # For manual review — leave blank
        })

        # Print progress
        status = f"  [{idx:3d}/{len(TEST_PROMPTS)}] {token_reduction:5.1f}% reduction | {prompt[:60]}..."
        print(status)

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    overall_reduction = (1 - total_comp_tokens / max(total_orig_tokens, 1)) * 100
    avg_time = sum(r["compression_time_ms"] for r in rows) / len(rows)

    print(f"\n{'='*80}")
    print(f"  BATCH TEST COMPLETE")
    print(f"{'='*80}")
    print(f"  Prompts tested     : {len(rows)}")
    print(f"  Lambda             : {args.lambda_val}")
    print(f"  Checkpoint         : {ckpt_path}")
    print(f"  Total orig tokens  : {total_orig_tokens}")
    print(f"  Total comp tokens  : {total_comp_tokens}")
    print(f"  Overall reduction  : {overall_reduction:.1f}%")
    print(f"  Avg time/prompt    : {avg_time:.1f}ms")
    print(f"  Output CSV         : {out_path}")
    print(f"{'='*80}")
    print(f"\n  Open the CSV and fill in 'quality_ok' (yes/no) and 'notes' columns for review.\n")


if __name__ == "__main__":
    main()
