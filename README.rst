https://pypi.org/project/seqtag/

BiLSTM + CRF for sequence tagging

This is adapted from guillaumegenthial 's original implementation and is made configurable and easy to adapt and use.

**Requirements:**

This code is tested with all tensorflow versions from 1.3.0 to 1.10.0.

tensorflow is not included in setup.py since, it will remove tensorflow-gpu.
Separately install tensorflow by following https://www.tensorflow.org/install/ for this module to work.

Download the 300 dimnesional glove vectors from https://nlp.stanford.edu/projects/glove/

**Installation:**

The stable version can be installed by running "pip install seqtag"

**How to Use:**

Create a training directory with train.txt and valid.txt (test.txt is optional)
set the config parameters as expected in a configuration file. 

An example of the configuration file can be found at https://github.com/bedapudi6788/seqtag/blob/master/example_config.json

**Training:**

from seqtag import trainer

trainer.train(config_path = 'path_to_config_json')

**Running Predictions:**

from seqtag import predictor

model = predictor.load_model(path_to_config.json)

predictor.predict(model, ['I', 'am', 'Batman'])

['O', 'O', 'B-PER']

For an usage example take a look at https://github.com/bedapudi6788/Deep-Segmentation/ . seqtag is used for sentence segmentation in this repo.
