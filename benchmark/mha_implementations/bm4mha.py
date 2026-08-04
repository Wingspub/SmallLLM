import torch
from torch.utils.benchmark import Timer
from typing import List

# ============== 因果注意力方法 ==================
from MHA_navie import navieMHA
from MHA_pytorch import MHA_pytorch
from MHA_pytorch_fa import MHA_pytorch_fa
from MHA_SPDA import MHA_SPDA
from MHA_SPDA_fa import MHA_SPDA_fa
# ======================================

device = torch.device("cuda")
# 基准数据测量
min_run_time = 30 # 秒
batch_size = 4
seq_len = 1024
embed_dims = 768

# 测量序列增长带来的性能差异

model_list: List = [
    navieMHA,
    MHA_pytorch,
    MHA_pytorch_fa,
    MHA_SPDA,
    MHA_SPDA_fa
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



