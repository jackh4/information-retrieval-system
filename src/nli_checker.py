from sentence_transformers import CrossEncoder
import torch

def load_nli_model(model_name='cross-encoder/nli-deberta-v3-base'):
    model = CrossEncoder(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

def predict_relation(model, sentence1, sentence2):
    scores = model.predict([sentence1, sentence2])
    label_mapping = ["contradiction", "entailment", "neutral"]
    predicted_label = label_mapping[scores.argmax()]
    return predicted_label