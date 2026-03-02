# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Détection et utilisation du GPU si disponible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f"Modèle initialisé sur: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU détecté: {torch.cuda.get_device_name(0)}")

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.as_tensor(np.asarray(state), dtype=torch.float).to(self.model.device)
        next_state = torch.as_tensor(np.asarray(next_state), dtype=torch.float).to(self.model.device)
        action = torch.as_tensor(np.asarray(action), dtype=torch.long).to(self.model.device)
        reward = torch.as_tensor(np.asarray(reward), dtype=torch.float).to(self.model.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                with torch.no_grad():
                    next_q_values = self.model(next_state[idx])
                    Q_new = reward[idx] + self.gamma * torch.max(next_q_values)

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()