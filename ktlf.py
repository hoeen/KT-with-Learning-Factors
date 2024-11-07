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

class KnowledgeLevel(Module):
    def __init__(self, num_learners, num_topics, num_kc, num_times):
        super(KnowledgeLevel, self).__init__()

        self.num_learners = num_learners
        self.num_topics = num_topics
        self.num_kc = num_kc
        self.num_times = num_times
        self.dim = 5

        self.epoch = 10 # TODO
        self.sigma_V = 0.1
        self.sigma2_R = 0.1

        self.neighbor_weights = Parameter(torch.randn((num_topics, num_topics)))

        self.V = Parameter(torch.randn((self.num_topics, self.num_kc), dtype=torch.float32))

        self.U_embedding = Linear(self.dim, 1)
        self.U = Parameter(torch.randn((self.num_times, self.num_learners, self.num_kc, self.dim), dtype=torch.float32))

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
            log_likelihood += log_prob.sum()
        
        return log_likelihood

    def train_model(
        self, data, optimizer
    ):
        Q, R, C, _, _, _, _ = data
        self.train()

        for epoch in range(1, self.epoch+1):
            mask = (Q.unsqueeze(1) * Q.unsqueeze(0)).bool().any(dim=-1)
            neighbor_weights = self.neighbor_weights.masked_fill(~mask, 0.0)

            V_new = torch.zeros_like(self.V)

            for j in range(self.num_topics):
                V_j_new = torch.zeros(self.num_kc, dtype=torch.float32)
                for d in range(mask.size(1)):
                    if mask[j,d]:
                        V_d = self.V[d, :]
                        V_j_new += neighbor_weights[j, d] * V_d

                theta_V = torch.normal(mean=0, std=self.sigma_V, size=(self.num_kc,))

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
        self.num_learners = num_learners
        self.num_kcs = num_kcs
        self.emb_size = 100
        self.hidden_size = 100
        self.epoch = 10 # TODO
        self.dim = 5

        self.alpha = Parameter(torch.rand((num_learners,)))
        self.lstm_layer = LSTM(
            self.num_kcs * 5, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_kcs * 5)
        self.U_embedding = Linear(self.dim, 1)
        self.sigmoid = Sigmoid()

        self.y = None
        
    def forward(self, U, V, Fshift, delta):
        
        deltashift = delta[1:]
        
        U = U[1:,:,:,:] # 시간 shift
        U_reshaped = U.view(-1, U.size(1), U.size(2) * U.size(3))
        h, _ = self.lstm_layer(U_reshaped)
        output = self.out_layer(h)
        U = output.view(U.size(0), U.size(1), U.size(2), U.size(3))
                    
        for t in range(U.size(0)-1): # 시간별
            for i in range(U.size(1)):
                for k in range(self.num_kcs):
                    memory_factor = self.alpha[i] * U[:-1][t, i, k] * (Fshift[t,i,k] / (Fshift[t,i,k] + 1e-5))
                    forgetting_factor = (1 - self.alpha[i]) * U[:-1][t, i, k] * torch.exp(-deltashift[t, i, k].clone().detach() / 1.0)
                    
                    U_new = memory_factor + forgetting_factor
                    U = U.clone()
                    U[t][i][k] = U_new
                    
        if self.y is None:
            self.y = torch.zeros((U.size(0), U.size(1), V.size(0)),
                                 requires_grad=True, device=U.device)
        y = torch.zeros_like(self.y) # test 에서 이것때문에 shape 안맞음

        for t in range(U.size(0)):
            for i in range(U.size(1)):
                for j in range(V.size(0)):
                    V_j = V[j]
                    U_t_i = self.U_embedding(U[t, i, :, :]) 

                    inner_product = U_t_i.squeeze(-1)@V_j
                    inner_product = self.sigmoid(inner_product)

                    y[t,i,j] = inner_product
                    
        return y
  
    def train_model(
        self, U, V, data, optimizer
    ):
        aucs = []
        loss_means = []

        max_auc = 0
        
        Q, R, C, F, N, M, T = data
        
        ### # 사용자 기준으로 Train / Test 쪼개기 # TODO
        train_users = math.floor(U.size(1) * 0.8)
        test_users = math.ceil(U.size(1) * 0.2)
        U_train = U[:,:train_users, :,:]
        U_test = U[:,-test_users:, :,:]
        
        T = F.size(0) - 1
        delta = torch.zeros(F.size(0), F.size(1), F.size(2))
        for t in range(F.size(0)):
            for i in range(U.size(1)):  # 각 학습자에 대해
                for k in range(U.size(2)):  # 각 지식 개념에 대해
                    for past in range(t-1, -1, -1):  # T-1 시점부터 역순으로
                        if F[past, i, k] > 0:
                            delta[t][i][k] = t - past
                            break
                        # 이전에 학습하지 않은 KC는 간격을 0으로 유지
                        else:
                            delta[t][i][k] = 0

        Fshift = F[1:] 
        F = F[:-1]
                            
        self.train()
        for i in range(1, self.epoch + 1):
            loss_mean = []

            y = self(U_train, V, Fshift[:,:train_users], delta[:,:train_users, :])
            
            mask = (R[1:, :train_users] != -1)
            
            y = torch.masked_select(y, mask)
            t = torch.masked_select(R[1:, :train_users], mask)

            # print("predict: ", y)
            # print("target: ", t)
            optimizer.zero_grad()
            loss = binary_cross_entropy(y, t)
            loss.backward()
            optimizer.step()

            loss_mean.append(loss.detach().cpu().numpy())
            
            predicted_labels = (y >= 0.5).float()
            target_labels = (t >= 0.5).float()
            acc = (predicted_labels == target_labels).sum().item() / t.numel()


            print(f"Epoch {i}/{self.epoch}, Loss: {loss.item()}, ACC: {acc}")
            
            

            with torch.no_grad():
                self.eval()


                y = self(U_test, V, Fshift[:,-test_users:], delta[:,-test_users:, :])

                test_y = y[:,:test_users]
                mask = (R[1:, -test_users:] != -1)

                y = torch.masked_select(test_y, mask)
                t = torch.masked_select(R[1:, -test_users:], mask)

                predicted_labels = (y >= 0.5).float()
                target_labels = (t >= 0.5).float()
                acc = (predicted_labels == target_labels).sum().item() / t.numel()
                
                loss_mean = np.mean(loss_mean) ## 에러
                print(f"TEST ACC: {acc}")
                
                
                # if auc > max_auc:
                #     torch.save(
                #         self.state_dict(),
                #         os.path.join(
                #             ckpt_path, "model.ckpt"
                #         )
                #     )
                #     max_auc = auc

                # aucs.append(auc)
                # loss_means.append(loss_mean)

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
    # print("U", U, U.shape)
    # print("V", V, V.shape)
    
    print("\n#### [Future Knowledge Prediction] ####")
    predict_model = FutureKnowledge(dataset.num_learners, dataset.num_kc)
    optimizer = Adam(predict_model.parameters(), lr=0.001)
    predict_model.train_model(U, V, data=(Q, R, C, F, learners, topics, times), optimizer=optimizer)