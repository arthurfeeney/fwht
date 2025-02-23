import fwht._fwht_triton
import torch
import triton

import fwht

DEVICE = 'cuda'

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["memcpy", "fwht"],
        line_names=["Mem Copy", "fwht"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="fwht-per-vs-memcpy",
        args={}
    ))

@triton.testing.perf_report(configs)
def benchmark(size, provider):
    a = torch.randn((16, 2048, size), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "memcpy":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.clone(a), quantiles=quantiles)
    if provider == 'fwht':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fwht._fwht_triton.fwht(a), quantiles=quantiles)
    return ms, max_ms, min_ms

benchmark.run(save_path='./', print_data=True)