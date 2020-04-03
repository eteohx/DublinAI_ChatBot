# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:10:56 2020

@author: Administrator
"""

import math
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nn_setup import EmbeddingNet, CyclicLR, cosine, batches

import torch
from torch import nn
from torch import optim

import pickle

##----------------------------------- LOAD DATA -----------------------------------------------###
# load ratings
ratings = pd.read_csv('G:/My Drive/DublinAI/Mini Projects/chatbot/the-movies-dataset/ratings.csv',low_memory=False)

# load movies
#metadata = pd.read_csv('./the-movies-dataset/metadata_prep.csv',low_memory=False)
#movies = metadata[['id','title','genres','overview','popularity','vote_average']]

def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()
    
    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)
    unique_movies = ratings.movieId.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, movie_to_index)

(n, m), (X, y), _ = create_dataset(ratings)
minmax = ratings.rating.min(), ratings.rating.max()

##----------------------------- TRAIN NEURAL NETWORK -----------------------------------------###

# Seed with same value to get reproducable results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)
RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

# Split to train and test set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}

net = EmbeddingNet(
    n_users=n, n_movies=m, 
    n_factors=300, hidden=[500, 500, 500], 
    embedding_dropout=0.05, dropouts=[0.5, 0.5, 0.25])


lr = 1e-3
wd = 1e-5
bs = 20000
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

net.to(device)
#criterion = nn.MSELoss(reduction='sum')
#optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
#iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
#scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))
#
#for epoch in range(n_epochs):
#    stats = {'epoch': epoch + 1, 'total': n_epochs}
#    print('epoch: '+ str(epoch+1))
#    
#    for phase in ('train', 'val'):
#        training = phase == 'train'
#        running_loss = 0.0
#        n_batches = 0
#        
#        for batch in batches(*datasets[phase], shuffle=training, bs=bs):
#            x_batch, y_batch = [b.to(device) for b in batch]
#            optimizer.zero_grad()
#        
#            # compute gradients only during 'train' phase
#            with torch.set_grad_enabled(training):
#                outputs = net(x_batch[:, 0], x_batch[:, 1], minmax)
#                loss = criterion(outputs, y_batch)
#                
#                # don't update weights and rates when in 'val' phase
#                if training:
#                    scheduler.step()
#                    loss.backward()
#                    optimizer.step()
#                    lr_history.extend(scheduler.get_lr())
#                    
#            running_loss += loss.item()
#            
#        epoch_loss = running_loss / dataset_sizes[phase]
#        stats[phase] = epoch_loss
#        
#        # early stopping: save weights of the best model so far
#        if phase == 'val':
#            if epoch_loss < best_loss:
#                print('loss improvement on epoch: %d' % (epoch + 1))
#                best_loss = epoch_loss
#                best_weights = copy.deepcopy(net.state_dict())
#                no_improvements = 0
#            else:
#                no_improvements += 1
#                
#    history.append(stats)
#    print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
#    if no_improvements >= patience:
#        print('early stopping after epoch {epoch:03d}'.format(**stats))
#        break
#
#
#ax = pd.DataFrame(history).drop(columns='total').plot(x='epoch')
#net.load_state_dict(best_weights)
#ground_truth, predictions = [], []
#
#with torch.no_grad():
#    for batch in batches(*datasets['val'], shuffle=False, bs=bs):
#        x_batch, y_batch = [b.to(device) for b in batch]
#        outputs = net(x_batch[:, 0], x_batch[:, 1], minmax)
#        ground_truth.extend(y_batch.tolist())
#        predictions.extend(outputs.tolist())
#
#ground_truth = np.asarray(ground_truth).ravel()
#predictions = np.asarray(predictions).ravel()
#
#final_loss = np.sqrt(np.mean((predictions - ground_truth)**2))
#print(f'Final RMSE: {final_loss:.4f}')
#
#with open('best.weights', 'wb') as file:
#    pickle.dump(best_weights, file)
#
#def to_numpy(tensor):
#    return tensor.cpu().numpy()
#
#_, _, (user_id_map, movie_id_map) = create_dataset(ratings)
#embed_to_original = [[v, k] for k, v in movie_id_map.items()]
#popular_movies = ratings.groupby('movieId').movieId.count().sort_values(ascending=False).values[:1000]
#
## ----------------------- SAVE MOVIE EMBEDDINGS ------------------------- #
#embed = to_numpy(net.m.weight.data)
#movie_id_map = np.asarray(embed_to_original)
#np.savetxt("movie_id_map.csv", movie_id_map, delimiter=",")
#np.savetxt("movie_embeddings.csv", embed, delimiter=",")

best_weights = pickle.load(open('G:/My Drive/DublinAI/Mini Projects/chatbot/nn/best.weights', 'rb'))
net.load_state_dict(best_weights)

ground_truth, predictions = [], []
a = torch.LongTensor(np.asarray(X_valid))
b = torch.FloatTensor(np.asarray(y_valid))

with torch.no_grad():
    for i in range(260):
        print(i)
        outputs = net(a[i*bs:(i+1)*bs, 0], a[i*bs:(i+1)*bs, 1], minmax)
        ground_truth.extend(b[i*bs:(i+1)*bs].tolist())
        predictions.extend(outputs.tolist())

ground_truth = np.asarray(ground_truth).ravel()
predictions = np.asarray(predictions).ravel()
final_loss = np.sqrt(np.mean((predictions - ground_truth)**2))
print(f'Final RMSE: {final_loss:.4f}')
