import joblib
from sentence_transformers import SentenceTransformer


def load_models(classifier_path: str, encoder_path: str):
    classifier = joblib.load(classifier_path)
    encoder = SentenceTransformer(encoder_path)

    return classifier, encoder


def predict(classifier, encoder, text: str) -> int:
    embedding = encoder.encode([text])
    prediction = classifier.predict(embedding)
    print(prediction)
    return prediction[0]
