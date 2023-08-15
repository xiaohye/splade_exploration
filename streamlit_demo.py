import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade

## -------- Function that calls the model and generates the prediction --------- ##
def getPredictions(doc, model_type_or_dir):
    # model_type_or_dir = "naver/splade_v2_max"
    # Original version: loading model and tokenizer from huggingface

    model = Splade(model_type_or_dir, agg="max")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    # now compute the document representation
    with torch.no_grad():
        doc_rep = model(d_kwargs=tokenizer(doc, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

    # get the number of non-zero dimensions in the rep:
    col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
    print("number of actual dimensions: ", len(col))

    # now let's inspect the bow representation:
    weights = doc_rep[col].cpu().tolist()
    d = {k: v for k, v in zip(col, weights)}
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    bow_rep = []
    for k, v in sorted_d.items():
        bow_rep.append((reverse_voc[k], round(v, 2)))
    # print("SPLADE BOW rep:\n", bow_rep)
    sorted_values = sorted(bow_rep, key=lambda x: x[1], reverse=True)
    return sorted_values


st.title('SPLADE Model')
st.write('Given a sentence, SPLADE outputs a list of relevant words with corresponding importance scores.')

model_type = st.selectbox(
    'Select a model',
    ('splade_v2_max', 'splade_v2_distil', 'splade-cocondenser-selfdistil', 'splade-cocondenser-ensembledistil'))


# st.write('You selected:', model_type)

doc = st.text_input('Input your query here:', 'original xbox games lot')
output = getPredictions(doc, "naver/"+model_type)

st.write('The generated word list is: ')
st.write(str(output))






