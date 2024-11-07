import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class Dataset(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.history = os.path.join(self.dataset_dir,'learning_record.csv')
        self.Q, self.R, self.C, self.F, self.learners, self.topics, self.times = self.preprocess()

        self.num_learners = len(self.learners)
        self.num_topics = len(self.topics)
        self.num_kc = self.Q.size(1)
        self.num_times = len(self.times)

    def preprocess(self):
        # TODO: 실제 데이터로 갈아끼우기
        history = pd.read_csv(self.history)

        # data = [
        #     {'Learner': 'u1', 'Topic': 'e2', 'Time': 't2', 'Score': 0.25},
        #     {'Learner': 'u1', 'Topic': 'e5', 'Time': 't3', 'Score': 0.0},
        #     {'Learner': 'u2', 'Topic': 'e1', 'Time': 't1', 'Score': 1.0},
        #     {'Learner': 'u2', 'Topic': 'e2', 'Time': 't2', 'Score': 0.0},
        #     {'Learner': 'u2', 'Topic': 'e4', 'Time': 't2', 'Score': 0.75},
        #     {'Learner': 'u3', 'Topic': 'e3', 'Time': 't3', 'Score': 0.0},
        # ]
        # history = pd.DataFrame(data)
        
        # TODO: 토픽(시험) - 스킬셋(KC) 맵핑 데이터 필요
        # Q = torch.tensor([
        #     [1, 0, 0, 0, 0],  # e1
        #     [0, 0, 0, 0, 1],  # e2
        #     [1, 0, 0, 0, 1],  # e3
        #     [0, 1, 1, 0, 0],  # e4
        #     [1, 0, 1, 1, 0],  # e5
        # ], dtype=torch.float32)
        
        learners = sorted(history['Learner'].unique())
        topics = sorted(history['Topic'].unique())
        times = sorted(history['Time'].unique())
        
        Q_df = pd.read_csv(os.path.join(self.dataset_dir, 'knowledge_incidence_matrix.csv'))
        Q = Q_df.values
        sort_index = [topics.index(item) for item in Q[:, 0]]
        sorted_arr = Q[np.argsort(sort_index)]
        Q = torch.tensor(sorted_arr[:,1:])

        num_learners = len(learners)
        num_topics = len(topics)
        num_kcs = Q.size(1)
        num_times = len(times)

        learner2index = {learner: idx for idx, learner in enumerate(learners)}
        topic2index = {topic: idx for idx, topic in enumerate(topics)}
        time2index = {time: idx for idx, time in enumerate(times)}

        # R_ij^t : The score of learner i on topic j at time t
        R = torch.full((history['Time'].nunique(), num_learners, num_topics), -1.0, dtype=torch.float32)
        # C : 학습 여부 행렬
        C = torch.zeros((num_times, num_learners, num_topics), dtype=torch.float32)

        for _, row in history.iterrows():
            time_idx = time2index[row['Time']]
            learner_idx = learner2index[row['Learner']]
            topic_idx = topic2index[row['Topic']]
            
            R[time_idx, learner_idx, topic_idx] = row['Score']
            C[time_idx, learner_idx, topic_idx] = 1
            
        # F: 빈도 계산 (토픽에 따른 KC 빈도)
        F = torch.zeros(num_times, num_learners, num_kcs)
        for t in tqdm(range(num_times)):
            time = times[t]
            time_df = history[history.Time == time]
            learners_at_time = time_df['Learner'].unique().tolist()
            for i, learner in enumerate(learners_at_time):
                ti_df = history[(history.Learner==learner) & (history.Time == time)]
                if(len(ti_df) == 0):
                    continue
                topic_index = [topics.index(target) for target in ti_df['Topic'].tolist()]
                f_t_i_k = torch.sum(Q[topic_index], dim=0)
                
                F[t, i] = f_t_i_k
                
        return Q, R, C, F, learners, topics, times