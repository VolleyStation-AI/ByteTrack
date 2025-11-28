import torch, time

device = "cuda:0"

M = 200
for N in [2048, 4096, 8192]:
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(M):
        c = a @ b
    torch.cuda.synchronize()
    t1 = time.time()
    print(N, f"time per matmul: {(t1 - t0) / M}")
