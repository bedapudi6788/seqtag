import os
import json
from . import data_builder
from .data_utils import CoNLLDataset
from .ner_model import NERModel
from .config import Config


def train(config_path, continue_training=False):

    config_params = json.load(open(config_path))

    # building data (creating word.txt, tags.txt, chars.txt)
    config_build_data = Config(**config_params, load=False)
    data_builder.build(config_build_data)

    # creating training config with load=True
    config_train = Config(**config_params, load=True)


    # build model
    model = NERModel(config_train)
    model.build()

    if continue_training:
        try:
            weights_path = os.path.join(config_params['dir_output'], 'model.weights/')
            print("Attempting to load weights from", weights_path)
            model.restore_session(weights_path)
            model.reinitialize_weights("proj")
            print("Restoring weights succesfull")
        except Exception as e:
            print("Restoring weights failed, Starting training from scratch")
            print(e)
            input()

    # create datasets
    dev   = CoNLLDataset(config_train.filename_dev, config_train.processing_word,
                         config_train.processing_tag, config_train.max_iter)
    train = CoNLLDataset(config_train.filename_train, config_train.processing_word,
                         config_train.processing_tag, config_train.max_iter)

    # train model
    model.train(train, dev)

    print("Trainig Complete!")
    print("Remove the events.tf files from the output directory if you don't need them. Note that removing them won't affect the predictions in anyway")

if __name__ == "__main__":
    print("Enter config json path")
    train(input())
