"""Measure token sizes of child and parent chunks."""
import json
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT/LLaMA-compatible tokenizer

# Child chunks
child_tokens = []
with open("data/chunks/child_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        chunk = json.loads(line)
        tokens = len(enc.encode(chunk["text"]))
        child_tokens.append(tokens)

# Parent chunks
parent_tokens = []
with open("data/chunks/parent_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        chunk = json.loads(line)
        tokens = len(enc.encode(chunk["text"]))
        parent_tokens.append(tokens)

print("=== Child Chunks ===")
print(f"  Count: {len(child_tokens)}")
print(f"  Min tokens: {min(child_tokens)}")
print(f"  Max tokens: {max(child_tokens)}")
print(f"  Avg tokens: {sum(child_tokens)/len(child_tokens):.1f}")
print(f"  Median tokens: {sorted(child_tokens)[len(child_tokens)//2]}")
print(f"  Total tokens: {sum(child_tokens):,}")

print(f"\n=== Parent Chunks ===")
print(f"  Count: {len(parent_tokens)}")
print(f"  Min tokens: {min(parent_tokens)}")
print(f"  Max tokens: {max(parent_tokens)}")
print(f"  Avg tokens: {sum(parent_tokens)/len(parent_tokens):.1f}")
print(f"  Median tokens: {sorted(parent_tokens)[len(parent_tokens)//2]}")
print(f"  Total tokens: {sum(parent_tokens):,}")

# Distribution buckets for child chunks
print(f"\n=== Child Token Distribution ===")
buckets = [(0,50),(50,100),(100,150),(150,200),(200,300),(300,400),(400,500),(500,1000)]
for lo, hi in buckets:
    count = sum(1 for t in child_tokens if lo <= t < hi)
    if count > 0:
        bar = "#" * (count * 40 // len(child_tokens))
        print(f"  {lo:4d}-{hi:4d}: {count:5d} ({count*100/len(child_tokens):5.1f}%) {bar}")
