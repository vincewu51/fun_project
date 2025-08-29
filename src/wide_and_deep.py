import torch
import torch.nn as nn
import torch.nn.functional as F

class WideAndDeep(nn.Module):
    def __init__(self, 
                 num_sparse_features,  # number of categorical features
                 num_sparse_unique_vals,  # list: num unique values for each cat feature
                 num_dense_features,  # number of continuous features
                 hidden_dims=[64, 32]):
        super().__init__()
        
        # ----- Deep part -----
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_vals, 8)  # embedding dim = 8
            for num_vals in num_sparse_unique_vals
        ])
        
        input_dim = 8 * num_sparse_features + num_dense_features
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.deep_mlp = nn.Sequential(*layers)
        
        # ----- Wide part -----
        self.wide = nn.Linear(num_sparse_features + num_dense_features, 1)
        
        # ----- Output -----
        self.output = nn.Linear(hidden_dims[-1] + 1, 1)  # combine wide + deep
    
    def forward(self, sparse_x, dense_x):
        """
        sparse_x: LongTensor [batch_size, num_sparse_features]
        dense_x: FloatTensor [batch_size, num_dense_features]
        """
        # Deep embeddings
        embed = [self.embeddings[i](sparse_x[:, i]) for i in range(len(self.embeddings))]
        embed = torch.cat(embed, dim=1)
        deep_input = torch.cat([embed, dense_x], dim=1)
        deep_out = self.deep_mlp(deep_input)
        
        # Wide linear
        wide_input = torch.cat([sparse_x.float(), dense_x], dim=1)
        wide_out = self.wide(wide_input)
        
        # Combine
        combined = torch.cat([deep_out, wide_out], dim=1)
        out = torch.sigmoid(self.output(combined))
        return out

# ---------------- Example usage ----------------
batch_size = 4
num_sparse_features = 3
num_sparse_unique_vals = [50, 20, 10]
num_dense_features = 2

model = WideAndDeep(num_sparse_features, num_sparse_unique_vals, num_dense_features)

sparse_x = torch.randint(0, 50, (batch_size, num_sparse_features))
dense_x = torch.randn(batch_size, num_dense_features)

output = model(sparse_x, dense_x)
print(output)
