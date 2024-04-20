import tqdm
from Bio import SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
from typing import Dict
from collections import Counter
import random
import obonet
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import gradio as gr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False) #.to(device)

# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

def get_embeddings(seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)

    return emb_0_per_protein

def predict(fasta_file):
    sequences = SeqIO.parse(fasta_file, "fasta")

    ids = []
    num_sequences=sum(1 for seq in sequences)
    embeds = np.zeros((num_sequences, 1024))
    i = 0
    with open(fasta_file, "r") as fastafile:
      # Iterate over each sequence in the file
      for sequence in SeqIO.parse(fastafile, "fasta"):
        # Access the sequence ID and sequence data
        seq_id = sequence.id
        seq_data = str(sequence.seq)
        embeds[i] = get_embeddings(seq_data).detach().cpu().numpy()
        print(embeds[i])
        ids.append(seq_id)
        i += 1
        
    INPUT_SHAPE=[1024]
    num_of_labels=1500

    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=num_of_labels, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', tf.keras.metrics.AUC()]
    )
    
    model.load_weights('./model5layer.weights.h5') #load model here
    labels_df=pd.read_csv('./labels.csv')
    labels_df=labels_df.drop(columns='Unnamed: 0')

    predictions = model.predict(embeds)
    predictions_list1=[]
    predictions_list2=[]

    # 'predictions' will contain the model's output for the custom input tensor
    # print(predictions)
    for prediction in predictions:
        tmp=[]
        t2=[]
        for i in prediction:
            x=0 if i<0.4 else 1
            tmp.append(x)
            t2.append(i)
        predictions_list1.append(tmp.copy())
        predictions_list2.append(t2.copy())

    label_columns = labels_df.columns

    # Convert the predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions_list1, columns=label_columns)
    p21=pd.DataFrame(predictions_list2, columns=label_columns)

    # Save the DataFrame to a CSV file
    predictions_df.to_csv("predictions.csv", index=False) #output csv
    p21.to_csv("decimal.csv",index=False)
    return "predictions.csv"

gr.Interface(
    predict,
    title = 'Multi-label Protein Function Prediction',
    inputs="file",
    outputs="file",
    description="Upload a fasta file containing protein sequence"
).launch(share=True,debug=True)
