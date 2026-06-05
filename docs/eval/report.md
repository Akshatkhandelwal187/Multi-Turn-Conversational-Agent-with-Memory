# Aria — Memory Ablation Report

Scenarios: **8** · Probes per strategy: **24**

| Strategy | Recall | Avg context tokens | Avg latency (ms) |
|---|---:|---:|---:|
| `no_memory` | 0% | 92 | 0.00 |
| `sliding_window` | 0% | 152 | 0.00 |
| `buffer` | 100% | 355 | 0.01 |
| `cognitive` | 96% | 209 | 0.24 |

**Takeaway:** cognitive memory recalls **96%** of planted facts — matching full-history *buffer* (100%) and beating the bounded *sliding window* (0%) — while using **41% fewer context tokens** than buffer (209 vs 355).
