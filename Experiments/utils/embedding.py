
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import torch
from gensim.models import Word2Vec as word2vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_random_walks(G, num_of_walk, length_of_walk):
  walks = list()
  for i in range(num_of_walk):
    node_list = list(G.nodes())
    for node in node_list:
      now_node = node
      walk = list()
      walk.append(str(node))
      for j in range(length_of_walk):
        next_node = random.choice(list(G.neighbors(now_node)))
        walk.append(str(next_node))
        now_node = node
      walks.append(walk)
  return walks

def get_emb_deepwalk(features, G, vector_size):
  # ランダムウォークを生成
  walks = make_random_walks(G, 20, 20)
  # gensim の Word2Vecを使った学習部分
  model = word2vec(walks, min_count=0, vector_size=vector_size, window=5, workers=1)

  size = features.shape[0]
  x = np.zeros((size,vector_size))

  cnt = 0
  for node in G.nodes():
    x[cnt] = model.wv[str(node)]
    cnt += 1

  #print(x)
  return x


def get_topk(features, mask, G, vector_size, top_num):
  x = get_emb_deepwalk(features, G, vector_size)
  size = features.shape[0]
  score = torch.zeros((size,size))

  for i in range(size):
    for j in range(size):
      if mask[j,0] == False:
        score[i,j] = -np.linalg.norm(x[i] - x[j],ord=2)
      else:
        score[i,j] = -float('inf')
  
  for i in range(size):
    score[i,i] = -float('inf')

  values, indices = torch.topk(score, top_num)
  #print(values)
  #print(indices)

  return indices

# 現状はstructのみに適応可能
def apply_embedding_mean(features, mask, dataset, miss_type, top_num=5):
  n_node = features.shape[0]
  n_feat = features.shape[1]

  topk = np.loadtxt('embedding/' + dataset + '.txt', delimiter=' ', dtype='int64') # (n_node, n_node)

  X = torch.zeros_like(features)
  if miss_type == "struct":
    for i in range(n_node):
      cnt = 0
      for j in range(n_node):
        if mask[topk[i,j],0] == False:
          X[i] += features[topk[i,j]]
          cnt += 1
          if cnt >= top_num:
            break
  else:
    for i in range(n_node):
      for j in range(n_feat):
        cnt = 0
        for k in range(n_node):
          if mask[topk[i,k],j] == False:
            X[i,j] += features[topk[i,k],j]
            cnt += 1
            if cnt >= top_num:
              break
  X /= top_num
  features[mask] = X[mask]
