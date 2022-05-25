from architectures.vit_standard import ViTStandard
from architectures.vit_for_small_datasets import ViTForSmallDatasets


def get_network(network_type, device):

    if network_type.lower() == 'vitstandard':
        model = ViTStandard(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            channels=3,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            device=device
        ).to(device)

    elif network_type.lower() == 'vit_for_small_datasets':
        model = ViTForSmallDatasets(
            image_size=256,
            patch_size=16,
            num_classes=2,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    else:
        raise NotImplementedError('Network type not supported. Only ViTStandard and ViTForSmallDatasets are supported.')

    return model
