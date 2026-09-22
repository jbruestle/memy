"""Throughput benchmark for the llama.cpp teacher server: C concurrent
/completion requests with ~P-token chat prompts and n_predict G, reports
aggregate generated tok/s and per-request latency. Run on an idle server
(no trainer sharing the GPU) to size the teacher for N trainers.

  python cloud/teacher_bench.py --url http://127.0.0.1:8080 --conc 8,32 --gen 512 --prompt 1500
  python cloud/teacher_bench.py --api vllm --conc 32,64          # vLLM OpenAI-compatible server
"""
import argparse, json, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

def post(url, path, payload, method="POST"):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode() if method == "POST" else None,
                                 method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--conc", default="8,32")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--prompt", type=int, default=1500, help="approx prompt tokens")
    ap.add_argument("--api", default="llamacpp", choices=["llamacpp", "vllm"])
    a = ap.parse_args()
    if a.api == "vllm":
        return main_vllm(a)
    props = post(a.url, "/props", {}, "GET")
    print("slots", props.get("total_slots"), "n_ctx/slot", props["default_generation_settings"]["n_ctx"])
    filler = ("The committee reviewed the quarterly logistics report and noted several discrepancies "
              "in the shipping manifests, which the auditors attributed to a clerical error. ")
    body = filler * 200
    toks = post(a.url, "/tokenize", {"content": body})["tokens"]
    ctx = post(a.url, "/detokenize", {"tokens": toks[:max(a.prompt - 60, 10)]})["content"]
    text = ("<|im_start|>user\nPlease read the following:\n" + ctx +
            "\n\nWrite a detailed, multi-paragraph summary and analysis of the above.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    prompt = post(a.url, "/tokenize", {"content": text})["tokens"]
    print("prompt tokens", len(prompt), "n_predict", a.gen)
    def one(i):
        t = time.time()
        r = post(a.url, "/completion", {"prompt": prompt, "n_predict": a.gen, "return_tokens": True,
                                        "cache_prompt": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                                        "min_p": 0.0, "repeat_penalty": 1.0, "seed": i})
        return time.time() - t, len(r["tokens"]), r.get("timings", {})
    for c in [int(x) for x in a.conc.split(",")]:
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(one, range(c)))
        wall = time.time() - t0
        gen = sum(n for _, n, _ in res)
        lat = [d for d, _, _ in res]
        pp = [t.get("prompt_per_second", 0) for _, _, t in res]
        print(f"conc {c:3d}: wall {wall:6.1f}s  gen {gen:6d} tok  {gen / wall:7.0f} tok/s aggregate  "
              f"prompt {c * len(prompt) / wall:7.0f} tok/s  latency med {statistics.median(lat):.1f}s max {max(lat):.1f}s  "
              f"per-slot prompt_pp {statistics.median(pp):.0f} tok/s")

def main_vllm(a):
    """Same measurement against vLLM: token-id prompts via /v1/completions,
    generated ids read back from logprobs with --return-tokens-as-token-ids."""
    from transformers import AutoTokenizer
    models = post(a.url, "/v1/models", {}, "GET")["data"]
    model = models[0]["id"]
    print("model", model, "max_model_len", models[0].get("max_model_len"))
    tok = AutoTokenizer.from_pretrained(model)
    filler = ("The committee reviewed the quarterly logistics report and noted several discrepancies "
              "in the shipping manifests, which the auditors attributed to a clerical error. ")
    ctx = tok.decode(tok(filler * 200)["input_ids"][:max(a.prompt - 60, 10)])
    text = ("<|im_start|>user\nPlease read the following:\n" + ctx +
            "\n\nWrite a detailed, multi-paragraph summary and analysis of the above.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    prompt = tok(text)["input_ids"]
    print("prompt tokens", len(prompt), "n_predict", a.gen)
    def one(i):
        t = time.time()
        r = post(a.url, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": a.gen,
                                            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": i,
                                            "logprobs": 0})
        c = r["choices"][0]
        ids = [int(t.split(":")[1]) for t in c["logprobs"]["tokens"]] if c.get("logprobs") else []
        return time.time() - t, r["usage"]["completion_tokens"], ids
    for c in [int(x) for x in a.conc.split(",")]:
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(one, range(c)))
        wall = time.time() - t0
        gen = sum(n for _, n, _ in res)
        lat = [d for d, _, _ in res]
        print(f"conc {c:3d}: wall {wall:6.1f}s  gen {gen:6d} tok  {gen / wall:7.0f} tok/s aggregate  "
              f"prompt {c * len(prompt) / wall:7.0f} tok/s  latency med {statistics.median(lat):.1f}s max {max(lat):.1f}s  "
              f"ids-returned {len(res[0][2]) == res[0][1]}")


if __name__ == "__main__":
    main()
