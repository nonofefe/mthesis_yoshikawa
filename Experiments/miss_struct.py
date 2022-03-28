import numpy as np
import torch
from utils import get_degree

class MissStruct:
    def __init__(self, mask, adj, split=1):
        self.mask = mask
        self.adj = adj
        self.split = split
        self.mask_node = self.calculate_mask_node()
        self.degree, self.mask_neighbor = self.calculate_mask_neighbor()

    def calculate_mask_node(self):
        return sum(self.mask.t(),0) / float(self.mask.shape[1])

    def calculate_mask_neighbor(self):
        mask_per = sum(self.mask.t(),0) / float(self.mask.shape[1])
        mask_per_neighbor = mask_per * 0
        #print(mask_per)
        deg_list = torch.zeros(mask_per.shape[0])
        adj_values = self.adj._values()
        ind_arr = self.adj._indices()
        n = self.adj._indices().size()[1]
        for i in range(n):
            node1 = ind_arr[0,i].item()
            node2 = ind_arr[1,i].item()
            deg_list[node1] += 1
            deg_list[node2] += 1
            mask_per_neighbor[node1] += mask_per[node2]
                
        for i in range(mask_per.shape[0]):
            deg_list[i] /= 2
        
        for i in range(self.adj.size()[0]):
            mask_per_neighbor[i] /= deg_list[i]
        return deg_list, mask_per_neighbor
