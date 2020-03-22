import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchsummary import summary


def display_configs():
    from fake_sentinel.pipeline.configs import CONFIGS

    print("{\n" + "\n".join("\t{!r}: {!r},".format(k, v) for k, v in CONFIGS.items()) + "\n}")


def display_training_curve(result_dir_path):
    history_path = Path(result_dir_path) / 'history.json'

    with history_path.open('r') as f:
        history = json.load(f)

    plt.plot(history['train'], 'b', label='train')
    plt.plot(history['val'], 'r', label='val')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def display_evaluation(eval_loss, eval_time):
    eval_loss, eval_time = float(eval_loss), float(eval_time)

    print('Log loss: {:.6f}'.format(eval_loss))
    print('Average evaluation time per video: {:.2f} seconds'.format(eval_time))


def display_model():
    from fake_sentinel.model.classifier import create_classifier

    model = create_classifier(pretrained=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _ = model.to(device)

    summary(model, (3, 299, 299))
