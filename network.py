from logic.vit_original import ViT
from linformer import Linformer
import torch


def get_network(network_type, device):

    if network_type.lower() == 'vit':
        model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            channels=3,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)

    elif network_type.lower() == 'densenet':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)

    else:
        raise NotImplementedError('Network type not supported. Only Vit and DenseNet are supported.')

    return model
