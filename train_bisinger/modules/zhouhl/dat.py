import torch
from torch import nn
import torch.nn.functional as F
import joblib
from modules.fastspeech.tts_modules import FFTBlocks

# https://zhuanlan.zhihu.com/p/75470256
# class grl_func(torch.autograd.Function):
#     def __init__(self):
#         super().__init__()

#     @ staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.save_for_backward(lambda_)
#         return x.view_as(x)

#     @ staticmethod
#     def backward(ctx, grad_output):
#         lambda_, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         return - lambda_ * grad_input, None


# class GRL(nn.Module):
#     def __init__(self, lambda_=0.):
#         super().__init__()
#         self.lambda_ = torch.tensor(lambda_)

#     def set_lambda(self, lambda_):
#         self.lambda_ = torch.tensor(lambda_)

#     def forward(self, x):
#         return grl_func.apply(x, self.lambda_)

# https://github.com/fungtion/DANN/blob/master/models/functions.py

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

    
class PreNet(nn.Module):
    '''
    inputs: [B, T, in]
    outputs: [B, T, hidden_size // 2]
    '''
    def __init__(self, inp_size, out_size, dropout_p=0.5):
        super().__init__()
        self.l1_inp_size = inp_size # 80
        self.l1_out_size = out_size * 2 # 256
        self.l2_out_size = out_size # 128

        self.linear1 = nn.Linear(self.l1_inp_size, self.l1_out_size)
        self.linear2 = nn.Linear(self.l1_out_size, self.l2_out_size)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
    def forward(self, x):
        # print(x.data.cpu().numpy())
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        return x

class Classifier(nn.Module):
    def __init__(self, inp_size, num_type):
        super().__init__()
        self.l1_inp_size = inp_size # 256
        self.l1_out_size = inp_size // 2 # 128
        self.l2_out_size = inp_size // 4 # 64
        self.l3_out_size = num_type

        self.linear1 = nn.Linear(self.l1_inp_size, self.l1_out_size)
        self.linear2 = nn.Linear(self.l1_out_size, self.l2_out_size)
        self.linear3 = nn.Linear(self.l2_out_size, self.l3_out_size)

    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x



class DAT(nn.Module):
    """
    Domain Adversarial Training module implementation according to paper Learn2Sing
    """
    def __init__(self, mel_dim, hidden_size, num_type):
        # [B, T, C]
        '''
        dur
        <class 'torch.Tensor'> torch.Size([12, 20])
        ================================================================
        mel2ph
        <class 'torch.Tensor'> torch.Size([12, 392])
        ================================================================
        decoder_inp
        <class 'torch.Tensor'> torch.Size([12, 392, 256])
        ================================================================
        mel_out
        <class 'torch.Tensor'> torch.Size([12, 392, 80])
        ================================================================
        '''
        super().__init__()
        self.inp_size = mel_dim
        self.hidden_size = hidden_size
        self.num_type = num_type
 
        self.prenet = PreNet(self.inp_size, self.hidden_size // 2)
        self.gru = nn.GRU(input_size=self.hidden_size // 2, hidden_size=self.hidden_size, batch_first=True)
        """
        rnn = nn.GRU(128, 256, batch_first=True)
        input = torch.randn(12, 392, 128)
        h0 = torch.randn(1, 12, 256)
        output, hn = rnn(input, h0)
        print(output.shape, hn.shape)
        # torch.Size([12, 392, 256]) torch.Size([1, 12, 256])
        """ 
        # self.grl = GRL()
        self.classifier = Classifier(self.hidden_size, self.num_type)

    def forward(self, x, dat_lambda):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        x = self.prenet(x)
        output, hidden = self.gru(x)
        hidden = hidden.squeeze(0)  # [1, B, hidden_size] ==> [B, hidden_size] 
        # print(f'hidden.shape = {hidden.shape}')   # torch.Size([12, 256])
        # hidden = self.grl(hidden)
        reverse_features = ReverseLayerF.apply(hidden, dat_lambda)
        y = self.classifier(reverse_features)
        return hidden, y


if __name__ == '__main__':
    mel_dim = 80
    hidden_size = 256
    num_type = 3

    # dat_wo = DAT(mel_dim, hidden_size, num_type, grl_flag=False).cuda()
    dat_with = DAT(mel_dim, hidden_size, num_type).cuda()
    ret = joblib.load('/Netdata/2022/zhouhl/ml_m4singer4.0/debug/fs2midi_ret_1')
    mel_out = ret['mel_out'].cuda()
    # labels = torch.Tensor([0,0,1,1,1,1,0,0,0,0,0,1]).long().cuda()
    labels = torch.empty(12, dtype=torch.long).random_(3).cuda()
    labels.requires_grad = False
    print(labels)
    loss = nn.CrossEntropyLoss()
    # print('****'*30)
    # print(dat_wo)
    # for name, parameters in dat_wo.named_parameters():
    #     print(name, ':', parameters.size())
    # print('****'*30)
    # print(dat_with)
    # for name, parameters in dat_with.named_parameters():
    #     print(name, ':', parameters.size())
    # print('****'*30)

    # print('===='*30)
    # y_wo = dat_wo(mel_out)
    # # print(f'y_wo.shape = {y_wo.shape}')   # torch.Size([12, 3])
    # # loss_wo = F.cross_entropy(y_wo, labels)
    # loss_wo = loss(y_wo, labels)
    # loss_wo.backward()
    # print(f'loss_wo = {loss_wo}')
    # print(f'dat_wo.classifier.linear1.weight.grad = {dat_wo.classifier.linear1.weight.grad}')
    # print(f'dat_wo.gru.weight_hh_l0.grad = {dat_wo.gru.weight_hh_l0.grad}')

    print('===='*30)
    _, y_with = dat_with(mel_out)
    # print(f'y_with.shape = {y_with.shape}')   # torch.Size([12, 3])
    # loss_with = F.cross_entropy(y_with, labels)
    loss_with = loss(y_with, labels)
    loss_with.backward()
    print(f'loss_with = {loss_with}')
    print(f'dat_with.classifier.linear1.weight.grad = {dat_with.classifier.linear1.weight.grad}')
    print(f'dat_with.gru.weight_hh_l0.grad = {dat_with.gru.weight_hh_l0.grad}')

        
    