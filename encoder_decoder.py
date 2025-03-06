import torch
from torch import nn
import lightning.pytorch as pl

from THP import get_non_pad_mask, Encoder, PAD

class LSTMPredictor(pl.LightningModule):
    def __init__(self, hidden_size, num_embeddings, num_layers=1):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(num_embeddings+1, hidden_size, padding_idx=PAD)
        self.lstm = nn.LSTM(input_size=hidden_size+1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.predictor = nn.Sequential(nn.Linear(hidden_size, 1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, types, times):
        type_embeds = self.embeddings(types)
        # print('type_embeds', type_embeds.shape, times.shape)
        outputs, (hidden, cell) = self.lstm(torch.cat([times.unsqueeze(-1), type_embeds], dim=2))
        pred = self.predictor(outputs.mean(dim=1))
        return pred
    
    def training_step(self, batch, batch_idx):
        pred = self.forward(batch[0], batch[2])
        bce = nn.BCEWithLogitsLoss()(pred, batch[3].unsqueeze(-1).float())
        self.log_dict({'total_val_loss': bce}, prog_bar=True, batch_size=len(batch), on_step=True, on_epoch=False)
        # return elbo - mll*self.mll_weight
        return bce

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch[0], batch[2])

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
class TransformerPredictor(pl.LightningModule):
    def __init__(self, num_types, d_model=256, d_rnn=128, d_inner=256, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.time_predictor = nn.Sequential(nn.ReLU6(), nn.Linear(d_model, 1))

    def forward(self, event_type, event_time):
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        time_prediction = self.time_predictor(enc_output.mean(dim=1))
        return time_prediction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch[0], batch[2])
    
    def training_step(self, batch, batch_idx):
        pred = self.forward(batch[0], batch[2].squeeze(-1))
        return nn.BCEWithLogitsLoss()(pred, batch[3].unsqueeze(-1).float())

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

class TransformerHawkes(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, num_types, d_model=256, d_rnn=128, d_inner=256, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.num_types = num_types
        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))
        # # OPTIONAL recurrent layer, this sometimes helps
        # self.rnn = RNN_layers(d_model, d_rnn)
        # prediction of next time stamp
        # self.time_predictor = Predictor(d_model, 1)
        self.time_predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU6(), nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.ReLU())
        # prediction of next event type
        # self.type_predictor = Predictor(d_model, num_types)
        # self.type_predictor = nn.Linear(d_model, num_types)
        self.type_predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU6(), nn.LayerNorm(d_model), nn.Linear(d_model, num_types))

    def encode(self, event_type, event_time):
        """
        Return the hidden representations.
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim.
        """
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        return enc_output
    
    def decode(self, enc_output):
        """
        Return the predictions.
        Input: enc_output: batch*seq_len*model_dim.
        Output: type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        time_prediction = self.time_predictor(enc_output)
        type_prediction = self.type_predictor(enc_output)
        all_hid = self.linear(enc_output)
        return type_prediction, time_prediction, all_hid

    def forward(self, event_type, event_time):
        """
        SHOULD NOT BE CALLED DIRECTLY.
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        
        non_pad_mask = get_non_pad_mask(event_type)

        # enc_output = self.encoder(event_type, event_time, non_pad_mask)
        # # enc_output = self.rnn(enc_output, non_pad_mask)
        # time_prediction = self.time_predictor(enc_output) * non_pad_mask
        # type_prediction = self.type_predictor(enc_output) * non_pad_mask

        # # print('time_prediction', time_prediction.shape)
        # # print(time_prediction)
        # # print('type_prediction', type_prediction.shape)
        # # print(type_prediction)
        
        # return enc_output, (type_prediction, time_prediction)

        enc_output = self.encode(event_type, event_time)
        type_prediction, time_prediction, all_hid = self.decode(enc_output)
        type_prediction = type_prediction * non_pad_mask
        time_prediction = time_prediction * non_pad_mask
        return enc_output, type_prediction, time_prediction, all_hid