import torch
from torch.utils.benchmark import Timer
from typing import List

# ============== 方法 ==================
from MHA_warpper import MHAwithWarpper

# ======================================

device = torch.device("cuda")
# 基准数据测量
min_run_time = 30 # 秒
batch_size = 4
seq_len = 1024
embed_dims = 768

# 测量序列增长带来的性能差异

model_list: List = [
    MHAwithWarpper,
]

embeddings = torch.rand((batch_size, seq_len, embed_dims)).to(device=device)
for MHA_Model in model_list:

    model = MHA_Model(
        dims=embed_dims,
        heads=12,
        p=0.0,
        if_bias=False
    ).to(device=device)

    timer = Timer(
        stmt="model(embeddings)",
        globals={"model": model, "embeddings": embeddings},
        label="多头注意力推理速度",
        description=f"b={batch_size}, seq_len={seq_len}, embed_dims={embed_dims}"
    )
    res = timer.blocked_autorange(min_run_time=min_run_time)
    print(res)



