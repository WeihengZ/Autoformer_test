from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

# define training function
def train(loader, model, optimizer, criterion, device, label_length):

    '''
    p.s. input is (batch, time_length, feature_dim)
         output is (batch, time_length, feature_dim)
    '''

    batch_loss = 0 
    for idx, (inputs, targets) in enumerate(tqdm(loader)):

        model.train()
        optimizer.zero_grad()
        inputs = inputs.to(device)  # (B,T,F)
        targets = targets.to(device)    # (B,T,F)
        dec_input = inputs[:,-label_length,:]
        if len(dec_input.shape) == 2:    # in the case of label_length is 1
            dec_input = dec_input.unsqueeze(1)
        outputs = model.forward(inputs, dec_input)     # [B,T,F]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        model.zero_grad()

        batch_loss += loss.detach().cpu().item()

    return batch_loss / (idx + 1)

@torch.no_grad()
def eval(loader, model, device, target_length, label_length):
    # batch_rmse_loss = np.zeros(12)
    batch_mae_loss = np.zeros(target_length)

    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        inputs = (inputs).to(device)  # (B,T,F)
        targets = targets.to(device)  # (B,T,F)
        dec_input = inputs[:,-label_length,:]
        if len(dec_input.shape) == 2:   # in the case of label_length is 1
            dec_input = dec_input.unsqueeze(1)
        outputs = model.forward(inputs, dec_input)     # [B,T,F]

        # calculate MAE of the prediction
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        for i in range(target_length):
            batch_mae_loss[i] += np.mean(np.abs(outputs[:,i,:] - targets[:,i,:]))

    print('mae loss:', batch_mae_loss / (idx + 1))

    return batch_mae_loss / (idx + 1)

@torch.no_grad()
def plotting(loader, model, device, label_length, feature_id, num_plots):

    '''
    feature_id: index of the feature
    num_plots: number of testpoint that we use
    '''

    '''
    plot the predicted value and ground truth value for specific feature
    loader here must have batch value of 1
    '''

    for idx, (inputs, targets) in enumerate(tqdm(loader)):
        model.eval()

        assert inputs.shape[0] == 1, 'test loader must have batch value == 1 for this function'

        inputs = (inputs).to(device)  # (1,T,F)
        targets = targets.to(device)  # (1,T,F)
        dec_input = inputs[:,-label_length,:]
        if len(dec_input.shape) == 2:   # in the case of label_length is 1
            dec_input = dec_input.unsqueeze(1)
        outputs = model.forward(inputs, dec_input)     # [1,T,F]

        # extract the predicted value and target value, (T)
        prediction = outputs[0,:,feature_id].cpu().detach().numpy()
        GT = targets[0,:,feature_id].cpu().detach().numpy()

        # make the plot
        fig = plt.figure()
        x = np.arange(np.size(GT))
        plt.plot(x, prediction, '-o', label='Prediction')
        plt.plot(x, GT, '-o', label='Ground truth')
        plt.xlabel('Prediction steps')
        plt.ylabel('y')
        plt.title('Feature_id == {}'.format(feature_id))
        plt.legend(loc=0)
        plt.grid()
        plt.savefig(r'./results/prediction_feature_{}_{}-th_testpoint.png'.format(feature_id, idx))
    
        if idx >= num_plots - 1:
            break

        

        