import os

import numpy as np
import torch
import math
from tqdm import tqdm

from torch.nn import Module, Parameter, Embedding, Linear, Sigmoid, LSTM, Dropout
from sklearn import metrics
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.optim import SGD, Adam

from data_loader import Dataset
import wandb

from utils import setSeeds


setSeeds()
wandb.login()
wandb.init(project="keit")

SATURATION_M = 6
SATURATION_H = 2


class KnowledgeLevel(Module):
    def __init__(self, num_learners, num_topics, num_kc, num_times):
        super(KnowledgeLevel, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("cuda is currently not available. Please check CUDA connection. Using CPU...")

        self.num_learners = num_learners
        self.num_topics = num_topics
        self.num_kc = num_kc
        self.num_times = num_times
        self.dim = 5

        self.epoch = 10 # TODO
        self.sigma_V = 0.1
        self.sigma2_R = 0.1

        self.neighbor_weights = Parameter(torch.randn((num_topics, num_topics))).to(self.device)

        self.V = Parameter(torch.randn((self.num_topics, self.num_kc), dtype=torch.float32)).to(self.device)

        self.U_embedding = Linear(self.dim, 1).to(self.device)
        self.U = Parameter(torch.randn((self.num_times, self.num_learners, self.num_kc, self.dim), dtype=torch.float32)).to(self.device)

        self.sigmoid = Sigmoid()


    def forward(self, Q, V, R, C):
        log_likelihood = 0.0
        
        V_j_matrix = (self.neighbor_weights.unsqueeze(-1) * V).sum(dim=0) + torch.normal(mean=0.0, std=self.sigma_V * torch.ones_like(V[0]))
        
        U_reshaped = self.U.view(-1, self.num_kc, self.dim)
        U_embedded = self.U_embedding(U_reshaped) 
        U_t_i_matrix = U_embedded.view(self.num_times, self.num_learners, self.num_kc, 1)
        
        
        C_mask = (C==1)
        C_indices = torch.where(C_mask)

        for t,i,j in zip(*C_indices):
            V_j = V_j_matrix[j]

            U_t_i = U_t_i_matrix[t, i, :, :]

            inner_product = U_t_i.squeeze(-1)@V_j
            inner_product = self.sigmoid(inner_product)
            
            normal_dist = torch.distributions.Normal(inner_product, self.sigma2_R)

            log_prob = normal_dist.log_prob(R[t, i, j])
            log_likelihood += log_prob.mean()
        
        return log_likelihood

    def train_model(
        self, data, optimizer
    ):
        Q, R, C, _, _, _, _ = data
        self.train()

        for epoch in range(1, self.epoch+1):
            mask = (Q.unsqueeze(1) * Q.unsqueeze(0)).bool().any(dim=-1).to(self.device)
            neighbor_weights = self.neighbor_weights.masked_fill(~mask, 0.0)

            V_new = torch.zeros_like(self.V)

            for j in range(self.num_topics):
                V_j_new = torch.zeros(self.num_kc, dtype=torch.float32).to(self.device)
                for d in range(mask.size(1)):
                    if mask[j,d]:
                        V_d = self.V[d, :]
                        V_j_new += neighbor_weights[j, d] * V_d

                theta_V = torch.normal(mean=0, std=self.sigma_V, size=(self.num_kc,)).to(self.device)

                V_new[j, :] = V_j_new + theta_V

            log_likelihood = self(Q, V_new, R, C) 
            
            optimizer.zero_grad()
            loss = - log_likelihood
            loss.backward()

            optimizer.step()
            
            print(f"Epoch {epoch}/{self.epoch}, Loss: {loss.item()}")
            
class FutureKnowledge(Module):
    def __init__(self, num_learners, num_kcs):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("cuda is currently not available. Please check CUDA connection. Using CPU...")
        self.num_learners = num_learners
        self.num_kcs = num_kcs
        self.emb_size = 100
        self.hidden_size = 100
        self.epoch = 130 # TODO
        self.dim = 5

        self.alpha = Parameter(torch.rand((num_learners,))).to(self.device)
        self.lstm_layer = LSTM(
            self.num_kcs * 5, self.hidden_size, batch_first=True
        ).to(self.device)
        self.out_layer = Linear(self.hidden_size, self.num_kcs * 5).to(self.device)
        self.U_embedding = Linear(self.dim, 1).to(self.device)
        self.sigmoid = Sigmoid().to(self.device)

        self.y = None
        
    def forward(self, U, V, Fshift, delta, alpha):
        
        deltashift = delta[1:].to(self.device)
        
        U_past = U[:-1].to(self.device)
        U_shift = U[1:].to(self.device)
        U_reshaped = U_shift.view(-1, U_shift.size(1), U_shift.size(2) * U_shift.size(3))
        h, _ = self.lstm_layer(U_reshaped)
        output = self.out_layer(h)
        U_shift = output.view(U_shift.size(0), U_shift.size(1), U_shift.size(2), U_shift.size(3))
                    
        Fshift = Fshift.to(self.device)
        alpha_reshaped = alpha.view(1,alpha.size(0),1,1)
        memory_factor = alpha_reshaped *U_past * SATURATION_M*(Fshift / (Fshift + SATURATION_H)).unsqueeze(-1)
        forgetting_factor = (torch.ones(alpha_reshaped.shape).to(self.device) - alpha_reshaped) * U_past * torch.exp(-deltashift.unsqueeze(-1) / 1.0)

        U_new = torch.zeros_like(U) 
        U_new[1:] = memory_factor + forgetting_factor
       
                    
        if self.y is None:
            self.y = torch.zeros((U_shift.size(0), U_new.size(1), V.size(0)),
                                 requires_grad=True, device=U.device)
        y = torch.zeros_like(self.y)


        U_t_i = self.U_embedding(U_new[1:, :, :, :]) 

        inner_product = U_t_i.squeeze(-1)@V.T
        y = self.sigmoid(inner_product)
                    
        return y
  
    def train_model(
        self, U, V, data, optimizer
    ):
        aucs = []
        loss_means = []

        max_auc = 0
        
        Q, R, C, F, N, M, T = data
        
        train_users = math.floor(U.size(1) * 0.8)
        test_users = math.ceil(U.size(1) * 0.2)
        U_train = U[:,:train_users, :,:]
        U_test = U[:,-test_users:, :,:]
        
        T = F.size(0) - 1
        F_mask = (F > 0)
        F_indices = torch.where(F_mask)

        delta = torch.zeros(F.size(0), F.size(1), F.size(2))
        
        for t,i,k in zip(*F_indices):
            for past in range(t-1, -1, -1): 
                if F_mask[past, i, k]:
                    delta[t][i][k] = t - past
                    break
                else:
                    delta[t][i][k] = 0

        Fshift = F[1:] 
        F = F[:-1]
                            
        self.train()
        for i in range(1, self.epoch + 1):
            loss_mean = []

            R = R.to(self.device)
            
            y = self(U_train, V, Fshift[:,:train_users], delta[:,:train_users, :], self.alpha[:train_users])
            
            mask = (R[1:, :train_users] != -1)
            
            y = torch.masked_select(y, mask)
            t = torch.masked_select(R[1:, :train_users], mask)

            optimizer.zero_grad()
            loss = binary_cross_entropy(y, t)
            loss.backward()
            optimizer.step()

            loss_mean.append(loss.detach().cpu().numpy())
            
            predicted_labels = (y >= 0.2).float()
            target_labels = (t >= 0.5).float()
            train_acc = (predicted_labels == target_labels).sum().item() / t.numel()

            print(f"Epoch {i}/{self.epoch}, Loss: {loss.item()}, ACC: {train_acc}")
            
            with torch.no_grad():
                self.eval()


                y = self(U_test, V, Fshift[:,-test_users:], delta[:,-test_users:, :], self.alpha[-test_users:])

                test_y = y[:,:test_users]
                mask = (R[1:, -test_users:] != -1)

                y = torch.masked_select(test_y, mask)
                t = torch.masked_select(R[1:, -test_users:], mask)

                predicted_labels = (y >= 0.2).float()
                target_labels = (t >= 0.5).float()
                test_acc = (predicted_labels == target_labels).sum().item() / t.numel()
                
                loss_mean = np.mean(loss_mean)
                print(f"TEST ACC: {test_acc}")
            

            wandb.log(
                {
                    "epoch": i,
                    "train_loss": loss,
                    "train_acc_epoch": train_acc,
                    "test_acc_epoch": test_acc,
                }
            )
            
        return aucs, loss_means

       
if __name__ == '__main__':
    
    dataset = Dataset("/home/datasets")
    Q, R, C, F, learners, topics, times = \
        dataset.Q, dataset.R, dataset.C, dataset.F, dataset.learners, dataset.topics, dataset.times

    print("#### [Learning Records Probability Maximum] ####")
    prob_model = KnowledgeLevel(dataset.num_learners, dataset.num_topics, dataset.num_kc, dataset.num_times)   
    
    optimizer = Adam(prob_model.parameters(), lr=0.001)
    prob_model.train_model(data=(Q, R, C, F, learners, topics, times), optimizer=optimizer)
    
    U = prob_model.U
    V = prob_model.V
    
    print("\n#### [Future Knowledge Prediction] ####")
    predict_model = FutureKnowledge(dataset.num_learners, dataset.num_kc)
    optimizer = Adam(predict_model.parameters(), lr=0.001)
    predict_model.train_model(U, V, data=(Q, R, C, F, learners, topics, times), optimizer=optimizer)