import fwht._fwht_triton
import torch
import triton

import fwht

DEVICE = 'cuda'

configs = []

for dtype in [torch.float16, torch.float32]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["size"],
            x_vals=[256, 512, 1024, 2048, 4096],
            line_arg="provider",
            line_vals=["memcpy", "fwht-func", "fwht-func-inplace", "fwht-module"],
            line_names=["Mem Copy", "fwht-func","fwht-func-inplace", "fwht-module"],
            styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("black", "-")],
            ylabel="ms",
            plot_name="fwht-per-vs-memcpy",
            args={"dtype": dtype}
        ))

@triton.testing.perf_report(configs)
def benchmark(size, dtype, provider):
    a = torch.randn((16, 2048, size), device=DEVICE, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "memcpy":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.clone(a), quantiles=quantiles)
    if provider == 'fwht-func':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fwht.fast_hadamard_transform(a), quantiles=quantiles)
    if provider == 'fwht-func-inplace':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fwht.fast_hadamard_transform(a, inplace=True), quantiles=quantiles)
    if provider == 'fwht-module':
        ht = fwht.FastHadamardTransform().to(a.device).to(a.dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ht(a), quantiles=quantiles)
    return ms, max_ms, min_ms

benchmark.run(save_path='./', print_data=True)