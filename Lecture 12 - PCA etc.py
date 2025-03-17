#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from tqdm import tqdm
import matplotlib.pyplot as plt


#%% Load the data
dataset = load_dataset('imdb', verification_mode='no_checks')


#%% Feature transform
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", 
                                  torch_dtype=torch.float16)

data = dataset['train'][::10]
batch_size = 100
vectors = []

with tqdm(total=len(data['text'])) as pbar:
    for i in range(0, len(data['text']), batch_size):
        encoded_input = tokenizer(data['text'][i:i+batch_size], truncation=True, 
                                  padding='max_length', max_length=10, 
                                  return_tensors='pt')
        output = model(**encoded_input)
        vectors.append(output.last_hidden_state[:, 0, :].detach().numpy())
        pbar.update(vectors[-1].shape[0])

X = np.concatenate(vectors)
y = np.asarray(data['label'])


#%% PCA
pca = PCA()
X_pca = pca.fit_transform(X)[:, :3]

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()


#%% tSNE
tsne = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3)
X_tsne = tsne.fit_transform(X)


#%% UMAP
umap = UMAP(n_components=3)
X_umap = umap.fit_transform(X)


#%% Plot
D = X_tsne
fig = plt.figure()
for i, (D, title) in enumerate(zip([X_pca, X_tsne, X_umap], ["PCA", "tSNE", "UMAP"])):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=y,  marker='.', cmap='coolwarm', alpha=.6)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
plt.show()

