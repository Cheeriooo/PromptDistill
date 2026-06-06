"""
Final stress test — adversarial and real-world edge cases.
"""
from __future__ import annotations
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from prompt_optimizer.model.optimizer_model import PromptOptimizerModel
from prompt_optimizer.training.config import TrainingConfig

cfg = TrainingConfig()
model = PromptOptimizerModel(cfg)
model.load("checkpoints/sweep_lambda_0_1/final.pt")
model.eval()

TESTS = [
    # === Original user-reported failures (regression guard) ===
    ("What is the capital of india", "qa", ["what", "capital", "india"]),
    ("Can you please provide me code for binary search", "code", ["code", "binary", "search"]),
    ("Could you please provide me with a comprehensive and detailed summary of the key events that led to the First World War?", "summarization", ["summary", "events", "first", "world", "war"]),

    # === Negation (previously failing) ===
    ("Write Python code that does not use recursion for factorial", "code", ["not", "recursion", "factorial"]),
    ("List 5 countries that are not in Europe", "qa", ["not", "europe", "countries"]),
    ("Don't use any global variables in the code", "code", ["global", "variables", "code"]),

    # === Adversarial: double negation ===
    ("It is not uncommon for neural networks to not converge", "qa", ["not", "neural", "networks", "converge"]),

    # === Adversarial: meaning-critical prepositions ===
    ("Convert temperature from Celsius to Fahrenheit", "qa", ["convert", "celsius", "fahrenheit"]),
    ("Translate this text from English to Spanish", "general", ["translate", "english", "spanish"]),

    # === Adversarial: very similar words that need context ===
    ("What is the difference between a compiler and an interpreter", "qa", ["what", "difference", "compiler", "interpreter"]),
    ("Explain the pros and cons of microservices versus monolith architecture", "qa", ["pros", "cons", "microservices", "monolith"]),

    # === Real-world verbose prompts ===
    ("Hey, I was just wondering if you could maybe help me out with something. I need to write a REST API in Python using Flask that handles CRUD operations for a todo list application", "code",
     ["rest", "api", "python", "flask", "crud", "todo"]),
    ("So basically what I need is for you to explain to me in very simple terms that even a child could understand how exactly does the internet work and what happens when you type a URL in the browser", "qa",
     ["how", "internet", "work", "url", "browser"]),

    # === Code with specific constraints ===
    ("Implement a thread-safe singleton pattern in Java using double-checked locking", "code",
     ["thread", "singleton", "java", "locking"]),
    ("Write a recursive function in C++ to calculate the Fibonacci sequence", "code",
     ["recursive", "function", "c", "fibonacci"]),

    # === Multi-part questions ===
    ("What are the advantages and disadvantages of using NoSQL databases compared to SQL databases", "qa",
     ["advantages", "disadvantages", "nosql", "sql", "databases"]),

    # === Proper nouns in unusual positions ===
    ("How did the Renaissance influence modern European art and culture", "qa",
     ["how", "renaissance", "european", "art", "culture"]),
    ("Explain the significance of the Magna Carta in British history", "qa",
     ["magna", "carta", "british", "history"]),

    # === Numbers + units ===
    ("What is the boiling point of water at 1 atmosphere of pressure", "qa",
     ["boiling", "water", "1", "atmosphere"]),
    ("How long does it take light to travel 1 light year", "qa",
     ["how", "light", "travel", "1", "light"]),

    # === Instructions with format specs ===
    ("Write a haiku about artificial intelligence", "creative",
     ["haiku", "artificial", "intelligence"]),
    ("Create a JSON schema for a user profile with name, email, and age fields", "code",
     ["json", "schema", "user", "profile", "name", "email", "age"]),

    # === Extremely short (should never over-compress) ===
    ("What is AI", "qa", ["what", "ai"]),
    ("Define love", "qa", ["define", "love"]),
    ("Hello world", "general", ["hello", "world"]),
]

print(f"\n{'='*120}")
print(f"  FINAL STRESS TEST — {len(TESTS)} adversarial + real-world cases")
print(f"{'='*120}\n")

pass_count = 0
fail_count = 0

for prompt, task, must_contain in TESTS:
    result = model.compress([prompt], [task])[0]
    result_lower = result.lower()
    
    orig_tok = len(model.token_scorer.tokenizer(prompt)["input_ids"])
    comp_tok = len(model.token_scorer.tokenizer(result)["input_ids"])
    reduction = (1 - comp_tok / orig_tok) * 100
    
    missing = [w for w in must_contain if w.lower() not in result_lower]
    passed = len(missing) == 0
    
    if passed:
        pass_count += 1
    else:
        fail_count += 1

    icon = "✅" if passed else "❌"
    status = "PASS" if passed else "FAIL"
    print(f"{icon} [{status}] ({reduction:4.0f}% red) {result}")
    if not passed:
        print(f"   INPUT  : {prompt}")
        print(f"   MISSING: {missing}")
    print()

print(f"{'='*120}")
print(f"  RESULTS: {pass_count}/{pass_count + fail_count} passed, {fail_count} failed")
print(f"{'='*120}")

sys.exit(0 if fail_count == 0 else 1)
