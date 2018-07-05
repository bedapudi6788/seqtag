__author__ = 'BEDAPUDI PRANEETH'

from model.ner_model import NERModel
from model.config import Config

config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)
print 'Loaded the model weights'

def preprocess_text(text):
    #add any preprocessing here
    preprocessed_text = text.strip().lstrip()
    preprocessed_text = preprocessed_text.split()
    return preprocessed_text

def predict_entities(text):
    words = preprocess_text(text)
    entity_list = model.predict(words)
    predicted_entities = {}
    for i, entity_name in enumerate(entity_list):
        if entity_name != 'O':
            try:
                predicted_entities[entity_name].append((words[i], i))
            except KeyError:
                predicted_entities[entity_name] = [(words[i], i)]
    return predicted_entities


if __name__ == '__main__':
    while True:
        print 'Enter text'
        print predict_entities(raw_input)()