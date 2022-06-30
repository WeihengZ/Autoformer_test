import pickle
import numpy as np
import random
import argparse
import torch
from torch import optim
import torch.nn as nn
from data import trainingset_construct
from train_test import train, eval
from models.Autoformer import Model as Model

# define arguments (copy from Autoformer)
if 'define_configs' == 'define_configs':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')

    # model define
    parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--enc_in', type=int, default=512, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=512, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=512, help='output size')
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=5, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    args = parser.parse_args()

# define device
device = torch.device('cuda:0')

# load the data
[train_data,t]=pickle.load(open('./data/Data_T5_IC1000.p','rb'))
train_data = np.asarray(train_data)
mean = np.mean(train_data)
std = np.std(train_data)
[test_data,t]=pickle.load(open('./data/Data_T5_IC100.p','rb'))
test_data = np.asarray(test_data)
train_loader = trainingset_construct(data=train_data, batch_val=48, input_length=args.seq_len, \
    target_length=args.pred_len, Shuffle=True, mean=mean, std=std)
test_loader = trainingset_construct(data=test_data, batch_val=12, input_length=args.seq_len, \
     target_length=args.pred_len, Shuffle=False, mean=mean, std=std)


# define the model, optimizer and the criterion
model = Model(configs=args).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# store the training history
training_epoch = 20
mae = np.zeros((training_epoch, args.pred_len))

# train and evaluate the model
for epoch in range(training_epoch):
    print('Current epoch:', epoch)
    
    # evaluate the MAE
    ma = eval(loader=test_loader, model=model, device=device, target_length=args.pred_len, label_length=args.label_len)
    mae[epoch, :] = ma

    np.save(r'./results/MAE_history.npy', mae)

    # train the model
    train_loss = train(loader=train_loader, model=model, optimizer=optimizer, criterion=criterion, device=device, label_length=args.label_len)

    print('training loss:', train_loss)

    torch.save(model.state_dict(), './model.pkl')

    