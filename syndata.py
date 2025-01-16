import numpy as np
import pandas as pd
import pickle
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import lightning.pytorch as pl
pl.seed_everything(0)
import os
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

from encoder_decoder import TransformerHawkes, LSTMPredictor
from data_processing import load_clinical_trial_datasets, process_patient_df, convert_dict_to_df
import THP


class PatientDataset(Dataset):
    def __init__(self, patients_list):
        self.patients_list = patients_list
    def __len__(self):
        return len(self.patients_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.patients_list[idx]

class EventDataModule(pl.LightningDataModule):
    def __init__(self, patients_list, batch_size=1):
        super().__init__()
        self.patient_dataset = PatientDataset(patients_list)
        self.batch_size = batch_size
        # self.train, self.val = random_split(self.patient_dataset, [.5, .5], generator=torch.Generator().manual_seed(42))
        self.train = self.val = self.patient_dataset
        print('train, val', len(self.train), len(self.val))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

class VAE(pl.LightningModule):
    def __init__(self, max_len, all_latent_dim, embedding_dim, num_embeddings, ts_representation_mode, end_token_ind):
        super().__init__()
        self.save_hyperparameters()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.all_latent_dim = all_latent_dim
        self.max_len = max_len
        self.end_token_ind = end_token_ind
        self.pretrain = False
        self.var_multiplier = 1.

        # if ts_representation_mode=='mlp':
        #     self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=all_latent_dim)
        #     self.ts_encoder = MLPEncoder(input_len=max_len, embedding_latent_dim=all_latent_dim)
        #     self.ts_decoder = MLPDecoder(input_len=max_len, embedding_latent_dim=all_latent_dim)
        # elif ts_representation_mode=='lstm':
        #     self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=all_latent_dim)
        #     self.ts_encoder = LSTMEncoder(input_size=2, hidden_size=all_latent_dim, num_layers=1)
        #     self.ts_decoder = LSTMDecoder(input_size=all_latent_dim, hidden_size=all_latent_dim, output_size=2, seq_len=max_len, num_layers=1)
        if ts_representation_mode=='hawkes_multivariate':
            self.ts_encoder = TransformerHawkes(num_types=num_embeddings, d_model=embedding_dim, n_layers=1)

            # self.decoder = nn.Sequential(nn.Linear(all_latent_dim, max_len*self.proj_dim, bias=False), nn.ReLU6())
            # self.default_events = nn.Parameter(torch.rand(all_latent_dim))
            self.decoder_linear = nn.Sequential(nn.ReLU6(), nn.Linear(all_latent_dim//max_len, embedding_dim), nn.ReLU6(), nn.LayerNorm(embedding_dim),
                                                nn.Linear(embedding_dim, embedding_dim), nn.ReLU6())
        else:
            raise NotImplementedError
        
        # distribution parameters
        self.fc_mu = nn.Sequential(nn.ReLU6(), nn.Linear(embedding_dim, all_latent_dim//max_len), nn.Tanh())
        self.fc_var = nn.Sequential(nn.ReLU6(), nn.Linear(embedding_dim, all_latent_dim//max_len), nn.ReLU6())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return torch.optim.Adam(self.parameters(), lr=1e-4)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.mean(axis=-1).mean()
        return kl

    def forward(self, batch):
        events = batch[0]
        values = batch[1]
        times = batch[2]

        patient_embedding = self.ts_encoder.encode(events, times)
        # timediff = torch.cat([torch.zeros_like(times[:,0]).unsqueeze(-1), times[:,1:] - times[:,:-1]], dim=-1).unsqueeze(-1)
        # patient_embedding = torch.cat([patient_embedding, timediff], dim=-1)

        # non_pad_mask = (events != 0).unsqueeze(-1).float()
        # patient_embedding = (patient_embedding*non_pad_mask).sum(dim=1)

        # encode x to get the mu and variance parameters
        mu = self.fc_mu(patient_embedding).view(-1, self.all_latent_dim)
        # mu = patient_embedding.view(-1, self.all_latent_dim+self.max_len)
        log_var  = self.fc_var(patient_embedding).view(-1, self.all_latent_dim)
        # print('mu', mu.shape, 'log_var', log_var.shape)

        # sample z from q
        # if use_mean:
        #     z = mu
        #     std = torch.zeros_like(z)
        # else:
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std * self.var_multiplier)
        z = q.rsample()

        # # #========== decoding process ==========
        # patient_embedding = self.decoder(z) + self.default_events
        # patient_embedding = patient_embedding.view(-1, self.max_len, self.proj_dim)
        enc_output = self.decoder_linear(z.view(-1, self.max_len, self.all_latent_dim//self.max_len))
        # enc_output = z.view(-1, self.max_len, self.all_latent_dim//self.max_len)

        # enc_weights = torch.matmul(patient_embedding, self.ts_encoder.encoder.event_emb.weight.T)
        # enc_weights = torch.softmax(enc_weights, dim=-1)
        # enc_output = torch.matmul(enc_weights, self.ts_encoder.encoder.event_emb.weight)
        # enc_output = self.decoder_linear2(patient_embedding) + enc_output
        # # enc_output = enc_output + patient_embedding
        # # print('enc_output', enc_output.shape)
        
        type_prediction, time_prediction, all_hid = self.ts_encoder.decode(enc_output)

        return type_prediction, time_prediction, all_hid, z, mu, std

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self.forward(batch)
    
    def training_step(self, batch, batch_idx):
        events = batch[0]
        values = batch[1]
        times = batch[2]

        non_pad_mask = (events != 0).unsqueeze(-1).float()

        type_prediction, time_prediction, all_hid, z, mu, std = self.forward(batch)
        # print('type_prediction', type_prediction.shape, 'time_prediction', time_prediction.shape, 'all_hid', all_hid.shape)
        # print('z', z.shape, 'mu', mu.shape, 'std', std.shape)
        # print('events', events.shape, 'times', times.shape)
        # print('non_pad_mask', non_pad_mask.shape)
        # # quit()
        events = (events.unsqueeze(-1)*non_pad_mask).squeeze(-1).long()
        times = (times.unsqueeze(-1)*non_pad_mask).squeeze(-1)
        type_prediction = type_prediction*non_pad_mask
        time_prediction = time_prediction*non_pad_mask
        all_hid = all_hid*non_pad_mask

        recon_loss = THP.time_loss(time_prediction, times, non_pad_mask)
        if not self.pretrain:
            kl = self.kl_divergence(z, mu, std)
        else:
            kl = 0

        elbo = (kl + recon_loss)

        event_ll, non_event_ll = THP.log_likelihood(all_hid, times, events, self.num_embeddings, self.ts_encoder.beta, self.ts_encoder.alpha)
        event_loss = -torch.mean(event_ll - non_event_ll)
        event_loss = event_loss / 100

        pred_loss, pred_num_event = THP.type_loss(type_prediction, events, nn.CrossEntropyLoss(ignore_index=-1, reduction='none'), non_pad_mask)

        END_token_indexes = (events == self.end_token_ind).nonzero(as_tuple=True)[0]
        END_token_loss = nn.CrossEntropyLoss(ignore_index=-1)(type_prediction[END_token_indexes].transpose(1, 2), events[END_token_indexes]-1)
        pred_loss = pred_loss + END_token_loss*10
        # print(pred_num_event)
        # pred_loss = pred_loss * 10
        # print('elbo', elbo.shape, 'recon_loss', recon_loss.shape, 'event_loss', event_loss.shape, 'pred_loss', pred_loss.shape)

        self.log_dict({
            'kl': kl,
            'recon_loss': recon_loss,
            'event_loss': event_loss,
            'pred_loss': pred_loss,
            'total_val_loss': elbo + event_loss + pred_loss
        }, prog_bar=True, batch_size=len(batch), on_step=True, on_epoch=False)
        # return elbo - mll*self.mll_weight
        # return (elbo + event_loss + pred_loss) / 1000
        return elbo + event_loss + pred_loss
        # return elbo 

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

def train_downstream_classifier(gen_df, df, indicator_col, value_num_col, patient_id_col, time_col, label_col, epochs=10, hidden_size=32):
    gen_patients_list = process_patient_df(gen_df, patient_id_col, indicator_col, value_num_col, time_col, label_col)
    data_list = process_patient_df(df, patient_id_col, indicator_col, value_num_col, time_col, label_col)
    
    val_patients_list, max_length, num_embeddings, events2ind, ind2events, mean_time, std_time = preprocess_data(data_list)
    print(events2ind)
    train_patients_list, max_length, num_embeddings, events2ind, ind2events, mean_time, std_time = preprocess_data(gen_patients_list, max_length=max_length, num_embeddings=num_embeddings, events2ind=events2ind, ind2events=ind2events, mean_time=mean_time, std_time=std_time)

    val_labels = [val_patients_list[i][3] for i in range(len(val_patients_list))]
    print('unique val_labels', np.unique(val_labels, return_counts=True))

    train_data = DataLoader(PatientDataset(train_patients_list), batch_size=8, shuffle=True, num_workers=1)
    val_data = DataLoader(PatientDataset(val_patients_list), batch_size=8, shuffle=False, num_workers=1)
    
    # print(num_events_to_generate)
    # downstream_classifier = TransformerPredictor(num_types=num_embeddings, d_model=hidden_size, n_layers=4)
    downstream_classifier = LSTMPredictor(num_embeddings=num_embeddings, hidden_size=hidden_size)
    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=True, logger=False, enable_checkpointing=False)
    trainer.fit(model=downstream_classifier, train_dataloaders=train_data)
    predictions = trainer.predict(model=downstream_classifier, dataloaders=val_data)
    
    predictions = torch.cat(predictions, dim=0).cpu().squeeze().numpy()
    rocauc = roc_auc_score(val_labels, predictions)
    prauc = average_precision_score(val_labels, predictions)
    preds_df = pd.DataFrame({'label': val_labels, 'pred': predictions})
    return {'preds_df': preds_df, 'rocauc': rocauc, 'prauc': prauc}

def preprocess_data(patients_list, max_length=None, num_embeddings=None, events2ind=None, ind2events=None, mean_time=None, std_time=None):
    # ================== data processing ==================
    for i in range(len(patients_list)):
        patients_list[i][0].append('[END]')
        patients_list[i][1].append(patients_list[i][1][-1])
        patients_list[i][2].append(patients_list[i][2][-1])

    if max_length is None:
        max_length = max([len(patients_list[i][0]) for i in range(len(patients_list))])
    
    unique_events = np.unique([_ for i in range(len(patients_list)) for _ in patients_list[i][0]])
    if num_embeddings is None:
        num_embeddings = len(unique_events) # padding is automatically done
    if events2ind is None:
        events2ind = {unique_events[i]: i for i in range(len(unique_events))} 
    if ind2events is None:
        ind2events = {v:k for k,v in events2ind.items()}
    if mean_time is None:
        mean_time = np.mean([np.mean(patients_list[i][2]) for i in range(len(patients_list))])
        std_time = np.mean([np.std(patients_list[i][2]) for i in range(len(patients_list))])

    for i in range(len(patients_list)):
        patients_list[i][0] = torch.Tensor([events2ind[_]+1 for _ in patients_list[i][0]] + [0]*(max_length-len(patients_list[i][0]))).long()
        patients_list[i][1] = torch.Tensor(patients_list[i][1] + [0]*(max_length-len(patients_list[i][1])))
        patients_list[i][2] = torch.Tensor(patients_list[i][2] + [0]*(max_length-len(patients_list[i][2])))
        patients_list[i][2] = patients_list[i][2] / std_time

    print('num_patient', len(patients_list), 'max_length', max_length)
    return patients_list, max_length, num_embeddings, events2ind, ind2events, 0, std_time

def run_experiment(df, indicator_col, value_num_col, patient_id_col, time_col, label_col, ts_representation_mode, save_dir,
                   checkpoint_path=None, train=True, var_multiplier=1.):
    patients_list = process_patient_df(df, patient_id_col, indicator_col, value_num_col, time_col, label_col)
    std_patients_list, max_length, num_embeddings, events2ind, ind2events, mean_time, std_time = preprocess_data(patients_list)

    datamodule = EventDataModule(patients_list=std_patients_list, batch_size=4)

    # ================== model training ==================
    autoencoder = VAE(max_len=max_length, all_latent_dim=32*max_length, embedding_dim=128, num_embeddings=num_embeddings, ts_representation_mode=ts_representation_mode,
                       end_token_ind=events2ind['[END]']+1)
    early_stopping = pl.callbacks.EarlyStopping('total_val_loss', patience=40, verbose=True, mode='min', check_on_train_epoch_end=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True, monitor='total_val_loss', mode='min')

    if checkpoint_path is not None:
        autoencoder = VAE.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # autoencoder.pretrain = True
    # trainer = pl.Trainer(max_epochs=5, enable_progress_bar=True, default_root_dir=None)
    # trainer.fit(model=autoencoder, datamodule=datamodule)

    trainer = pl.Trainer(max_epochs=80, enable_progress_bar=True, default_root_dir=save_dir, callbacks=[early_stopping, checkpoint_callback])
    if train:
        trainer.fit(model=autoencoder, datamodule=datamodule)
        trainer.save_checkpoint(save_dir+'final_model.ckpt')
    autoencoder.var_multiplier = var_multiplier
    predictions = trainer.predict(model=autoencoder, dataloaders=datamodule.predict_dataloader())
    
    # # ================== convert to gen_patients_list ==================
    all_type_predictions = torch.cat([_[0] for _ in predictions], dim=0)
    all_time_predictions = torch.cat([_[1] for _ in predictions], dim=0)
    all_type_predictions = torch.softmax(all_type_predictions.cpu(), dim=-1).numpy()
    # print('all_type_predictions', all_type_predictions.shape, 'all_time_predictions', all_time_predictions.shape)

    all_time_predictions = np.cumsum(all_time_predictions.cpu().numpy(), axis=1) * std_time + mean_time

    gen_patients_list = []
    for i in range(len(all_type_predictions)):
        inds = np.arange(all_type_predictions.shape[-1])
        # all_type_predictions_i = [np.random.choice(inds, p=all_type_predictions[i,j]) 
        #                           for j in range(len(all_type_predictions[i]))]
        all_type_predictions_i = np.argmax(all_type_predictions[i], axis=-1)
        type_prediction = [ind2events[_] for _ in all_type_predictions_i]
        # print(all_type_predictions_i)
        # print(type_prediction)
        if '[END]' in type_prediction:
            type_prediction_ = type_prediction[:type_prediction.index('[END]')]
            time_prediction_ = list(all_time_predictions[i][:len(type_prediction_)])
        else:
            type_prediction_ = type_prediction[:max_length-1]
            time_prediction_ = list(all_time_predictions[i][:max_length-1])
        # print('type_prediction', len(type_prediction_), 'time_prediction', len(time_prediction_))
        gen_patients_list.append([
            type_prediction_, # events
            [0]*len(type_prediction_), # values
            time_prediction_, # times
            patients_list[i][3] # label
        ])

    gen_df = convert_dict_to_df(gen_patients_list, indicator_col, value_num_col, patient_id_col, time_col, label_col)    
    print('gen_df', gen_df.head())
    # print('predictions', predictions)
    return gen_df

if __name__=='__main__':
    # ================== model training ==================
    parser = argparse.ArgumentParser()
    parser.add_argument('-run', type=int)
    parser.add_argument('-mode')
    parser.add_argument('-ts_representation_mode', default='hawkes_multivariate')
    parser.add_argument('-log_dir', type=str, default='./lightning_logs_new/{}_{}/')
    parser.add_argument('-var_multiplier', type=float, default=1.)
    args = parser.parse_args()
    assert args.run in [1,2,3,4,5,6,7]
    # assert args.mode in ['train', 'test', 'privacy','sanity_test']

    # data_list, orig_df_list = load_clinical_trial_datasets(trial_path='./data/')
    prefix_list = ['NCT00003299', 'NCT00041119', 'NCT00079274', 'NCT00174655', 'NCT00312208', 'NCT00694382', 'NCT03041311']
    data_args = load_clinical_trial_datasets(trial_path='../data/', dataset_to_process=[prefix_list[args.run-1]])
    data_args = data_args[0]
    save_dir = args.log_dir.format(prefix_list[args.run-1], args.ts_representation_mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('================== {} {} =================='.format(prefix_list[args.run-1], args.ts_representation_mode))
    if args.mode == 'train':
        gen_df = run_experiment(df=data_args['df'], 
                                           indicator_col=data_args['indicator_col'], 
                                           value_num_col=data_args['value_num_col'], 
                                           patient_id_col=data_args['patient_id_col'], 
                                           time_col=data_args['time_col'], 
                                           label_col=data_args['label_col'], 
                                           ts_representation_mode=args.ts_representation_mode, 
                                           save_dir=save_dir)
        gen_df.to_csv(save_dir+'gen_df.csv', index=False)
        data_args['df'].to_csv(save_dir+'orig_df.csv', index=False)
        # with open(os.path.join(save_dir, 'preds.pkl'), 'wb') as f:
        #     pickle.dump(gen_patients_list, f)
        # also dump dataframe

    elif args.mode == 'test':
        # with open(os.path.join(save_dir, 'preds.pkl'), 'rb') as f:
        #     gen_data_list = pickle.load(f)
        gen_df = pd.read_csv(save_dir+'gen_df.csv')
        gen_df[data_args['indicator_col']] = gen_df[data_args['indicator_col']].astype(str)
        gen_df[data_args['value_num_col']] = gen_df[data_args['value_num_col']].astype(float)
        gen_df[data_args['patient_id_col']] = gen_df[data_args['patient_id_col']].astype(str)
        gen_df[data_args['time_col']] = gen_df[data_args['time_col']].astype(float)
        gen_df[data_args['label_col']] = gen_df[data_args['label_col']].astype(int)
        downstream_results = train_downstream_classifier(gen_df, data_args['df'], 
                                           indicator_col=data_args['indicator_col'], 
                                           value_num_col=data_args['value_num_col'], 
                                           patient_id_col=data_args['patient_id_col'], 
                                           time_col=data_args['time_col'], 
                                           label_col=data_args['label_col'], epochs=10, hidden_size=64)
        downstream_results['preds_df'].to_csv(save_dir+'downstream_preds_roc={}_prauc={}.csv'.format(downstream_results['rocauc'], downstream_results['prauc']), index=False)
        print('preds {}, roc_auc {}, pr_auc {}'.format(downstream_results['preds_df'].shape, downstream_results['rocauc'], downstream_results['prauc']))

    elif args.mode == 'privacy_ablation':
        latest_version_path = f'{save_dir}/final_model.ckpt'

        for var_multiplier in [0.1, 0.5, 1., 1.25, 1.5, 1.75, 2., 3., 4.]:
            gen_df = run_experiment(df=data_args['df'], 
                                           indicator_col=data_args['indicator_col'], 
                                           value_num_col=data_args['value_num_col'], 
                                           patient_id_col=data_args['patient_id_col'], 
                                           time_col=data_args['time_col'], 
                                           label_col=data_args['label_col'], 
                                           ts_representation_mode=args.ts_representation_mode, 
                                           save_dir=save_dir, checkpoint_path=latest_version_path, train=False, var_multiplier=var_multiplier)
            downstream_results = train_downstream_classifier(gen_df, data_args['df'],
                                                indicator_col=data_args['indicator_col'], 
                                                value_num_col=data_args['value_num_col'], 
                                                patient_id_col=data_args['patient_id_col'], 
                                                time_col=data_args['time_col'], 
                                                label_col=data_args['label_col'], epochs=10, hidden_size=64)
            print('preds {}, roc_auc {}, pr_auc {}'.format(downstream_results['preds_df'].shape, downstream_results['rocauc'], downstream_results['prauc']))
            downstream_results['preds_df'].to_csv(save_dir+'sanity_roc_var_multiplier={}={}_prauc={}.csv'.format(var_multiplier, downstream_results['rocauc'], downstream_results['prauc']), index=False)  
                        
            gen_df.to_csv(save_dir+f'gen_df_var_multiplier={var_multiplier}.csv', index=False)
            print('var_multiplier', var_multiplier)


    elif args.mode == 'sanity_test':
        downstream_results = train_downstream_classifier(data_args['df'], data_args['df'],
                                             indicator_col=data_args['indicator_col'], 
                                             value_num_col=data_args['value_num_col'], 
                                             patient_id_col=data_args['patient_id_col'], 
                                             time_col=data_args['time_col'], 
                                             label_col=data_args['label_col'], epochs=10, hidden_size=64)
        downstream_results['preds_df'].to_csv(save_dir+'sanity_roc={}_prauc={}.csv'.format(downstream_results['rocauc'], downstream_results['prauc']), index=False)  
        print('preds {}, roc_auc {}, pr_auc {}'.format(downstream_results['preds_df'].shape, downstream_results['rocauc'], downstream_results['prauc']))
    else:
        raise NotImplementedError