import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, seq_len, pred_len):
        super(TimeSeriesTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, model_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(model_dim * seq_len, output_dim * pred_len)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding

        x = self.transformer_encoder(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        x = self.softmax(x)

        return x