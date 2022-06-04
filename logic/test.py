import tqdm

import torch
import torch.nn as nn

from utils import calculate_accuracy
from dataloader import get_test_loader
from architectures.densenet3d import DenseNet


def test(model, test_loader):

    test_accuracy = 0

    with torch.no_grad():
        for data, label in tqdm.tqdm(test_loader, "Testing patches"):
            data = data.to(torch.float32)
            label = label.to(torch.float32)

            test_output = model(data)
            acc = calculate_accuracy(test_output, label)
            test_accuracy += acc / len(test_loader)

    print(f"Test accuracy: {test_accuracy:.4f}")


# load model
model = DenseNet()
model_params = torch.load("/Users/fryderykkogl/Data/temp/model_epoch8_valacc0.932.pt",
                          map_location=torch.device('cpu'))
model.load_state_dict(model_params['model_state_dict'])
model.eval()

# load test data
loader = get_test_loader("/Users/fryderykkogl/Data/temp/test_data05")

test(model, loader)
