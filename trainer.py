import torch
import torch.nn as nn
import torch.nn.functional as F
from util import calc_rmse_reg, calc_mae_reg,calc_rmse_mae_reg,common_loss, loss_dependence

class Trainer:
    def __init__(self, model, data, calc_rmse_mae, optimizer, alpha, regression):
        self.model = model
        # self.dataset = dataset
        self.alpha = alpha
        self.regression = regression
        self.tau: float = 0.5
        self.mean: bool = True
        self.data = data
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.calc_rmse_mae = calc_rmse_mae
        # self.calc_rmse = calc_rmse
        # self.calc_mae = calc_mae
        self.optimizer = optimizer
        # self.experiment = experiment

    def training(self, epochs):
        self.epochs = epochs
        max_val = 5
        min_val = 1
        best_test_rmse = 100
        best_test_mae = 100
        loss_fn = nn.MSELoss()
        for epoch in range(1,epochs+1):
            # if epoch % 50 == 0:
            #     self.alpha_loss = self.alpha_loss * 0.8
            loss, train_rmse = self.train_one(max_val,min_val,loss_fn)
            test_rmse, test_mae = self.test()
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse

            if test_mae < best_test_mae:
                best_test_mae = test_mae

            self.summary(epoch, loss, train_rmse, test_rmse, best_test_rmse, test_mae, best_test_mae)

        print('END TRAINING')
        return best_test_rmse,best_test_mae


    def train_one(self,max_val,min_val,loss_fn):
        self.model.train()
        u_emb_v1, i_emb_v1, u_emb_v2, i_emb_v2, u_emb_v3, i_emb_v3, u_emb_v4, i_emb_v4, out = self.model(self.data.edge_index_ui,
                                                                 self.data.S, self.data.S_dict,self.data.edge_index_uu,
                                                                 self.data.C, self.data.C_dict,self.data.edge_index_ii,
                                                                 self.data.UOCU, self.data.UOCU_dict,self.data.edge_index_uocu,
                                                                 self.data.IGEI, self.data.IGEI_dict,self.data.edge_index_igei)
        if self.regression == True:
            label_ = (self.data.y[self.train_mask]) / (max_val - min_val) + 0.01
            loss1 = loss_fn(out[self.train_mask].squeeze(1),label_)
        else:
            loss1 = F.cross_entropy(out[self.train_mask], self.data.y[self.train_mask])

        #自监督损失
        # loss_dep = (loss_dependence(u_emb_v1, u_emb_v2, self.data.num_users) + loss_dependence(i_emb_v1, i_emb_v2, self.data.num_items))/2
        #user间自监督
        u_l1 = self.semi_loss(u_emb_v3, u_emb_v4)
        u_l2 = self.semi_loss(u_emb_v4, u_emb_v3)
        u_ret = (u_l1 + u_l2) * 0.5
        u_ret = u_ret.mean() if self.mean else u_ret.sum()
        #item间自监督
        i_l1 = self.semi_loss(i_emb_v3, i_emb_v4)
        i_l2 = self.semi_loss(i_emb_v4, i_emb_v3)
        i_ret = (i_l1 + i_l2) * 0.5
        i_ret = i_ret.mean() if self.mean else i_ret.sum()

        #总损失
        loss = loss1 + self.alpha * (u_ret + i_ret)
        # loss = loss1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.regression == True:
            rmse, _ = calc_rmse_mae_reg(out[self.train_mask], self.data.y[self.train_mask])
        else:
            rmse, _ = self.calc_rmse_mae(out[self.train_mask], self.data.y[self.train_mask])
        return loss.item(), rmse.item()

    def test(self):
        self.model.eval()
        out = self.model(self.data.edge_index_ui,
                         self.data.S, self.data.S_dict, self.data.edge_index_uu,
                         self.data.C, self.data.C_dict, self.data.edge_index_ii,
                         self.data.UOCU, self.data.UOCU_dict, self.data.edge_index_uocu,
                         self.data.IGEI, self.data.IGEI_dict, self.data.edge_index_igei)[8]

        if self.regression == True:
            rmse, mae = calc_rmse_mae_reg(out[self.test_mask], self.data.y[self.test_mask])
            # mae = calc_mae_reg(out[self.test_mask], self.data.y[self.test_mask])
        else:
            rmse, mae = self.calc_rmse_mae(out[self.test_mask], self.data.y[self.test_mask])
            # mae = self.calc_mae(out[self.test_mask], self.data.y[self.test_mask])
        return rmse.item(), mae.item()

    def summary(self, epoch, loss, train_rmse=None, test_rmse=None, best_test_rmse=None,test_mae=None, best_test_mae=None):
        if test_rmse is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} | Best Test RMSE: {:.6f} | Test MAE: {:.6f} | Best Test MAE: {:.6f} | Alpha: {:.6f} ]'.format(
                    epoch, self.epochs, loss, train_rmse, test_rmse, best_test_rmse, test_mae, best_test_mae, self.alpha))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        l = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # l = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) + between_sim.diag()))
        # l1 = -torch.log(between_sim.diag() / (refl_sim.sum(1)+ between_sim.diag())) #视图内损失
        # l2 = -torch.log(between_sim.diag() / between_sim.sum(1))   #视图间损失
        # l =  l1 + l2

        return l