from logic.vit import ViT
from linformer import Linformer
import torch


def get_network(params, device):

    if params.network_type.lower() == 'vit':
        model = ViT(
            dim=int(params.network_dim),
            image_size=224,
            patch_size=int(params.patch_size),
            num_classes=2,
            channels=3,
            depth=int(params.depth),
            heads=int(params.heads),
            mlp_dim=int(params.mlp_dim),
            dropout=0.1,
            emb_dropout=0.1,
            device=device
        ).to(device)

    elif params.network_type.lower() == 'densenet':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)

    else:
        raise NotImplementedError('Network type not supported. Only Vit and DenseNet are supported.')

    return model
