__author__ = 'BEDAPUDI PRANEETH'
import json
from .ner_model import NERModel
from .config import Config

def load_model(config_path):
    config_params = json.load(open(config_path))
    config = Config(**config_params, load=True)
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    print("Completed loading the model")
    return model

def predict(model, in_sequence = []):
    pred_sequence = model.predict(in_sequence)
    return pred_sequence

if __name__ == '__main__':
    pass
