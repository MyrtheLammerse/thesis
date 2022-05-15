from rasa.nlu.components import Component
from transformers import pipeline


class SentimentAnalyzer(Component):
    """A pre-trained sentiment component"""

    name = "sentiment"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["en"]

    # initialize the huggingface classifier
    sentiment_labels = ['enjoyment', 'fear', 'sadness', 'anger']
    classifier = pipeline("zero-shot-classification")

    def __init__(self, component_config=None):
        super(SentimentAnalyzer, self).__init__(component_config)

    def train(self, training_data, cfg, **kwargs):
        """Not needed, because the the model is pretrained"""
        pass

    def convert_to_rasa(self, value, confidence):
        """Convert model output into the Rasa NLU compatible output format."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

        return entity

    def process(self, message, **kwargs):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""
        try:
            res = self.classifier(message.data['text'], self.sentiment_labels)
            key, value = res['labels'][0], res['scores'][0]
            entity = self.convert_to_rasa(key, value)
            message.set("entities", [entity], add_to_output=True)
        except KeyError:
            pass

    def sentiment(self, message):
        """Retrieve the text message, pass it to the classifier
            and append the prediction results to the message class."""
        entity = self.classifier(message, self.sentiment_labels)
        return entity

    def persist(self, file_name, dir_name):
        """Pass because a pre-trained model is already persisted"""
        pass
