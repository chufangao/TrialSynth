# Original Source: https://github.com/SimiaoZuo/Transformer-Hawkes-Process
# MIT License

# Copyright (c) 2024 Simiao Zuo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
PAD = 0

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(all_hid, time, non_pad_mask, type_mask, alpha, beta):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * \
                torch.rand([*diff_time.size(), num_samples], device=all_hid.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = all_hid[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + alpha * temp_time, beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(all_hid, time, types, num_types, beta, alpha):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), num_types], device=all_hid.device)
    for i in range(num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(all_hid.device)

    all_lambda = softplus(all_hid, beta)
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(all_hid, time, non_pad_mask, type_mask, alpha, beta)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func, non_pad_mask):
    """ Event prediction loss, cross entropy or label smoothing. """

    # print(prediction.shape, types.shape)
    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth) / non_pad_mask.sum()

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    # print(loss.shape, non_pad_mask.shape)
    loss = torch.sum(loss*non_pad_mask[:,:-1].squeeze(), dim=-1).mean()
    return loss, correct_num


def time_loss(prediction, event_time, non_pad_mask):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    # true = event_time[:, 1:] - event_time[:, :-1]
    # prediction = prediction[:, :-1]
    true = event_time
    prediction = torch.cumsum(prediction, dim=1)
    # event time gap prediction
    diff = prediction - true
    se = torch.sum((diff * diff) * non_pad_mask.squeeze(), dim=-1).mean()
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


def get_non_pad_mask(seq):
    """ Get the non-padding positions.
    e.g. seq shape of (batch_size, seq_len) -> return (batch_size, seq_len, 1) # seq_len varies
    seq.shape = torch.Size([4, 5])
    tensor([[2, 2, 2, 0, 0],
            [7, 7, 8, 7, 7],
            [2, 2, 2, 2, 0],
            [2, 2, 0, 0, 0]], device='cuda:0')

    out.shape = torch.Size([4, 5, 1])
    tensor([[[1.], [1.], [1.], [0.], [0.]],
            [[1.], [1.], [1.], [1.], [1.]],
            [[1.], [1.], [1.], [1.], [0.]],
            [[1.], [1.], [0.], [0.], [0.]]], device='cuda:0')
    """

    # print("get_non_pad_mask")
    # print(seq.shape)
    # print(seq)
    # print(seq.ne(PAD).type(torch.float).unsqueeze(-1).shape)
    # print(seq.ne(PAD).type(torch.float).unsqueeze(-1)); quit()
    assert seq.dim() == 2
    return (seq!=PAD).type(torch.float).unsqueeze(-1).to(seq.device)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. 
    seq_k.shape = torch.Size([4, 5])
    tensor([[ 7,  7,  7,  0,  0],
            [15, 15, 15, 15, 16],
            [12, 11, 12, 12,  0],
            [ 1,  1,  1,  0,  0]], device='cuda:0')
    seq_q.shape = torch.Size([4, 5])
    tensor([[ 7,  7,  7,  0,  0],
            [15, 15, 15, 15, 16],
            [12, 11, 12, 12,  0],
            [ 1,  1,  1,  0,  0]], device='cuda:0')
    out.shape = torch.Size([4, 5, 5])
    tensor([[[False, False, False,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False,  True,  True]],

            [[False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False]],

            [[False, False, False, False,  True],
            [False, False, False, False,  True],
            [False, False, False, False,  True],
            [False, False, False, False,  True],
            [False, False, False, False,  True]],

            [[False, False, False,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False,  True,  True]]], device='cuda:0')
    """
    # print("get_attn_key_pad_mask")
    # print(seq_k.shape)
    # print(seq_k)
    # print(seq_q.shape)
    # print(seq_q)

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = (seq_k == PAD).unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. 
    seq.shape = torch.Size([4, 5])
    tensor([[ 1,  1,  1,  0,  0],
            [ 9,  9,  9,  0,  0],
            [ 2,  2,  2,  2,  2],
            [33, 33, 33,  0,  0]], device='cuda:0')
    out.shape = torch.Size([4, 5, 5])
    tensor([[[0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]],

            [[0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]],

            [[0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]],

            [[0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]]], device='cuda:0', dtype=torch.uint8)
    """

    # print(seq.shape)
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    # print("get_subsequent_mask")
    # print(seq.shape)
    # print(seq)
    # print(subsequent_mask.shape)
    # print(subsequent_mask)
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, num_types, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, device='cuda'):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor([
            math.pow(10000.0, 2.0 * (i // 2) / d_model)
            for i in range(d_model)
            ],device=torch.device(device)) #size = (d_model,)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: (batch,seq_len)
        Output: (batch,seq_len,d_model)
        """
        # print('time.shape', time.shape)
        # print(time)
        # print('position_vec.shape', self.position_vec.shape)
        # print(self.position_vec)
        # quit()

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. 
        event_type.shape = torch.Size([4, 6])
        tensor([[ 2,  9,  2,  9,  9,  2],
                [ 1,  1,  1,  0,  0,  0],
                [ 2,  2,  2,  0,  0,  0],
                [34, 34,  0,  0,  0,  0]], device='cuda:0')
        event_time.shape = torch.Size([4, 6])
        tensor([[0.0000, 0.1731, 0.2885, 0.9808, 1.3654, 1.9231],
                [0.0000, 0.0577, 0.2115, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.2500, 1.2885, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.4423, 0.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')
        non_pad_mask.shape = torch.Size([4, 6, 1])
        tensor([[[1.], [1.], [1.], [1.], [1.], [1.]],
                [[1.], [1.], [1.], [0.], [0.], [0.]],
                [[1.], [1.], [1.], [0.], [0.], [0.]],
                [[1.], [1.], [0.], [0.], [0.], [0.]]], device='cuda:0')
        out.shape = torch.Size([4, 6, 512])
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq) > 0

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        # print("event_type.shape", event_type.shape)
        # print("event_time", event_type)
        # print("enc_output.shape", enc_output.shape)
        # print("tem_enc.shape", tem_enc.shape)
        # quit()
        enc_output += tem_enc

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output
class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, num_types, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, device='cuda'):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor([
            math.pow(10000.0, 2.0 * (i // 2) / d_model)
            for i in range(d_model)
            ],device=torch.device(device)) #size = (d_model,)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: (batch,seq_len)
        Output: (batch,seq_len,d_model)
        """
        # print('time.shape', time.shape)
        # print(time)
        # print('position_vec.shape', self.position_vec.shape)
        # print(self.position_vec)
        # quit()

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. 
        event_type.shape = torch.Size([4, 6])
        tensor([[ 2,  9,  2,  9,  9,  2],
                [ 1,  1,  1,  0,  0,  0],
                [ 2,  2,  2,  0,  0,  0],
                [34, 34,  0,  0,  0,  0]], device='cuda:0')
        event_time.shape = torch.Size([4, 6])
        tensor([[0.0000, 0.1731, 0.2885, 0.9808, 1.3654, 1.9231],
                [0.0000, 0.0577, 0.2115, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.2500, 1.2885, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.4423, 0.0000, 0.0000, 0.0000, 0.0000]], device='cuda:0')
        non_pad_mask.shape = torch.Size([4, 6, 1])
        tensor([[[1.], [1.], [1.], [1.], [1.], [1.]],
                [[1.], [1.], [1.], [0.], [0.], [0.]],
                [[1.], [1.], [1.], [0.], [0.], [0.]],
                [[1.], [1.], [0.], [0.], [0.], [0.]]], device='cuda:0')
        out.shape = torch.Size([4, 6, 512])
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq) > 0

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        # print("event_type.shape", event_type.shape)
        # print("event_time", event_type)
        # print("enc_output.shape", enc_output.shape)
        # print("tem_enc.shape", tem_enc.shape)
        # quit()
        enc_output += tem_enc

        # print(enc_output.shape, non_pad_mask.shape, slf_attn_mask.shape)
        # quit()
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


# class RNN_layers(nn.Module):
#     """
#     Optional recurrent layers. This is inspired by the fact that adding
#     recurrent layers on top of the Transformer helps language modeling.
#     """

#     def __init__(self, d_model, d_rnn):
#         super().__init__()

#         self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
#         self.projection = nn.Linear(d_rnn, d_model)

#     def forward(self, data, non_pad_mask):
#         lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
#         pack_enc_output = nn.utils.rnn.pack_padded_sequence(
#             data, lengths, batch_first=True, enforce_sorted=False)
#         temp = self.rnn(pack_enc_output)[0]
#         out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

#         out = self.projection(out)
#         return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, num_types, d_model=256, d_rnn=128, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
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
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        # enc_output = self.rnn(enc_output, non_pad_mask)
        time_prediction = self.time_predictor(enc_output, non_pad_mask)
        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        # print('time_prediction', time_prediction.shape)
        # print(time_prediction)
        # print('type_prediction', type_prediction.shape)
        # print(type_prediction)
        
        return enc_output, (type_prediction, time_prediction)
