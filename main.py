import os
import webbrowser
from transformers import CamembertConfig
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import CamembertForSequenceClassification
from torchinfo import summary
from captum.attr import (
    visualization
)
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Transformer_Explainability.BERT_explainability.modules.BERT.BERT import RobertaSelfAttention


def ensure_compatibility2(state_dict1, state_dict2):
    l1_compat = {}
    for item1, item2 in zip(state_dict1.items(), state_dict2.items()):
        key, value = item1
        if key != item2[0]:
            l1_compat[key.replace("roberta", "bert")] = value
        else:
            l1_compat[key] = value

    l1_compat["classifier.weight"] = state_dict1["classifier.out_proj.weight"]
    l1_compat["classifier.bias"] = state_dict1["classifier.out_proj.bias"]

    for elem in l1_compat:
        print(elem)

    return l1_compat

def viz_text():
    # Use a breakpoint in the code line below to debug your script.
    #model_bert_expl0 = CamembertForSequenceClassification.from_pretrained(r'C:\Users\jerem\OneDrive\Bureau\Pro\CamemBert-LRP\camembert_model')
    #model_bert = AutoModelForSequenceClassification.from_pretrained(r'C:\Users\jerem\OneDrive\Bureau\Pro\CamemBert-LRP\camembert_model')
    model_bert_tok = AutoTokenizer.from_pretrained(r'C:\Users\jerem\OneDrive\Bureau\Pro\CamemBert-LRP\camembert_model')

    #compat_dict = ensure_compatibility2(model_bert.state_dict(), model_bert_expl0.state_dict())

    model_bert_expl = CamembertForSequenceClassification.from_pretrained(r'C:\Users\jerem\OneDrive\Bureau\Pro\CamemBert-LRP\camembert_model')
    model_bert_expl.eval()

    explanations = Generator(model_bert_expl)

    classifications = ["NEGATIVE", "POSITIVE"]
    # encode a sentence
    text_batch = ["Premier chiffre: 242 milliards d’euros.Telle est la somme invraisemblable qui repose sur les "
                  "comptes d’épargne en Belgique. Si on a la curiosité de diviser ce chiffre par dix millions de "
                  "Belges, ça nous fait tout de même une moyenne de 24.000 € par personne. Si, comme moi, "
                  "vous avez l’impression de connaître davantage d’individus en-dessous des 24.000 qu’au-dessus, "
                  "sachez que vous faites partie de la mauvaise moyenne. De toute façon, bonne chance pour aborder le "
                  "sujet: l’argent qu’on gagne, l’argent qu’on thésaurise est et restera le dernier tabou des "
                  "Belges. Mais lorsque sortent ces chiffres, comme la semaine dernière, se pose logiquement la "
                  "question de la manière dont cet argent pourrait être mobilisé et utilisé. C’est vrai que de "
                  "l’argent fixe, ça n’arrange personne: ni l’Etat, qui se ponctionne lorsque l’argent circule par "
                  "la TVA, l’IPP, l’ISOC, etc., ni les entreprises, ni même les particuliers, à qui il ne rapporte "
                  "rien."]
    encoding = model_bert_tok(text_batch, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # true class is positive - 1
    true_class = 1

    # generate an explanation for the input
    expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
    # normalize scores
    expl = (expl - expl.min()) / (expl.max() - expl.min())

    # get the model classification
    output = torch.nn.functional.softmax(model_bert_expl(input_ids=input_ids, attention_mask=attention_mask)[1], dim=-1)
    classification = output.argmax(dim=-1).item()
    # get class name
    class_name = classifications[classification]
    # if the classification is negative, higher explanation scores are more negative
    # flip for visualization
    if class_name == "NEGATIVE":
        expl *= (-1)

    tokens = model_bert_tok.convert_ids_to_tokens(input_ids.flatten())
    #print(output)
    #print([(tokens[i], expl[i].item()) for i in range(len(tokens))])
    vis_data_records = [visualization.VisualizationDataRecord(
        expl,
        output[0][classification],
        classification,
        true_class,
        true_class,
        0,
        tokens,
        1)]
    visualization.visualize_text(vis_data_records)
    print([(tokens[i], expl[i].item()) for i in range(len(tokens))])



if __name__ == '__main__':
    viz_text()

