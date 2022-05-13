from vit_pytorch.efficient import ViT
from linformer import Linformer


def get_network(device="cpu"):
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )

    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=2,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)

    return model
