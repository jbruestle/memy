"""Parity of MemoryReader._gathered (training top-K) against _explicit (diagnostics
top-K): outputs, grads, stats. Run with FP32=1 for exact agreement (bf16 differs
only through the explicit path keeping ties at the K-th score); the last line
reports the gathered path's extra memory at v2 sizes (T=4k, N=24k).

  FP32=1 python test_topk.py
"""
import torch, math, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from model import MemoryReader, MemoryContext
torch.manual_seed(0)
dev = 'cuda'; import os; DT = torch.float32 if os.environ.get('FP32') else torch.bfloat16
for (B,T,N,K,masked) in [(2,37,300,8,True),(3,1100,2500,8,True),(1,16,5,8,False),(2,64,64,4,True)]:
    rd = MemoryReader(256, n_heads=4, rank=32, layer_idx=0).to(dev).to(DT)
    torch.nn.init.normal_(rd.mem_o.weight, std=0.02)
    h = torch.randn(B,T,256, device=dev, dtype=DT, requires_grad=True)
    m = torch.randn(B,N,256, device=dev, dtype=DT, requires_grad=True)
    mask = torch.ones(B,N, device=dev)
    if masked:
        mask[0, N//2:] = 0
    ctx = MemoryContext(); ctx.memory=m; ctx.memory_mask=mask; ctx.top_k=K; ctx.log_stats=True
    outs, grads, stats = [], [], []
    for use_explicit in (True, False):
        ctx.capture_maps = use_explicit; ctx.maps=[]; ctx.stats=[]
        h.grad=None; m.grad=None; rd.zero_grad()
        out = rd(h, ctx)
        out.float().pow(2).sum().backward()
        outs.append(out.detach().float()); grads.append((h.grad.float().clone(), m.grad.float().clone(), rd.mem_q.weight.grad.float().clone()))
        stats.append(ctx.stats[-1])
    d = lambda a,b: ((a-b).abs().max() / (b.abs().max()+1e-9)).item()
    print(f"B{B} T{T} N{N} K{K} masked={masked}: out rel {d(outs[1],outs[0]):.2e}  grad_h {d(grads[1][0],grads[0][0]):.2e}  grad_m {d(grads[1][1],grads[0][1]):.2e}  grad_q {d(grads[1][2],grads[0][2]):.2e}  entropy {stats[0]['entropy']:.4f} vs {stats[1]['entropy']:.4f}  argmax_pos {stats[0]['argmax_mean_pos']:.1f} vs {stats[1]['argmax_mean_pos']:.1f}")
# memory check at v2 size
B,T,N,K = 4,4096,24000,8
rd = MemoryReader(2560, n_heads=4, rank=128, layer_idx=0).to(dev).to(DT)
torch.nn.init.normal_(rd.mem_o.weight, std=0.02)
h = torch.randn(B,T,2560, device=dev, dtype=DT, requires_grad=True)
m = torch.randn(B,N,2560, device=dev, dtype=DT, requires_grad=True)
ctx = MemoryContext(); ctx.memory=m; ctx.memory_mask=torch.ones(B,N,device=dev); ctx.top_k=K
torch.cuda.reset_peak_memory_stats(); base=torch.cuda.memory_allocated()
out = rd(h, ctx); out.float().sum().backward()
print(f"v2-size gathered fwd+bwd (B{B} T{T} N{N} K{K}): peak extra {(torch.cuda.max_memory_allocated()-base)/1e9:.2f} GB")
