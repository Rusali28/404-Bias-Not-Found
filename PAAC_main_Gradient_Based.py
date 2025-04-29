import argparse
import os.path
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import scipy as scipy

import os
import metrics, dataloader, utils,mini_batch_test
import ast

Tmin = 0.05  # for popular items (sharper)
Tmax = 1.0 # for unpopular items (softer)

def main_args():

    # model description
    args = argparse.ArgumentParser(description="PAAC")

    # dataset
    args.add_argument('--dataset_name', default='gowalla', type=str)

    ##OOD
    args.add_argument('--dataset_path', default='OOD_Data', type=str)
    args.add_argument('--result_path', default='OOD_result', type=str)

    args.add_argument('--bpr_num_neg', default=1, type=int)

    # LightGCN model
    args.add_argument('--model', default='PAAC', type=str)
    args.add_argument('--decay', default=0.0001, type=float)
    args.add_argument('--lr', default=0.001, type=float)
    args.add_argument('--batch_size', default=2048, type=int)
    args.add_argument('--layers_list', default='[3]', type=str)
    args.add_argument('--eps', default=0.2, type=float)
    args.add_argument('--cl_rate_list', default='[0.2]', type=str)
    args.add_argument('--temperature_list', default='[0.2]', type=str)
    args.add_argument('--seed', default=12345, type=int)
    args.add_argument('--align_reg_list', default='[1]', type=str)
    args.add_argument('--lambada_list', default='[0.2]', type=str)
    args.add_argument('--gama_list', default='[0.2]', type=str)
    #args.add_argument('--align_reg_list', default='[100]', type=str)

    # train
    args.add_argument('--device', default=0, type=int)
    args.add_argument('--EarlyStop', default=10, type=int)
    args.add_argument('--emb_size', default=64, type=int)
    args.add_argument('--num_epoch', default=30, type=int)
    

    args.add_argument(
        '--topks', default='[20]', type=str)

    return args.parse_args()


class PAAC(torch.nn.Module):
    def __init__(self, config, data):
        super(PAAC, self).__init__()
        self.emb_size = config.emb_size
        self.decay = config.decay
        self.layers = config.layers
        self.device = torch.device(config.device)
        self.eps = config.eps
        self.cl_rate = config.cl_rate
        self.temperature = config.temperature
        self.pop_train = data.pop_train_count
        self.lambda2 = config.lambda2
        self.gamma = config.gamma
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.adj = data.norm_adj.to(self.device)

        user_emb_weight = torch.nn.init.normal_(torch.empty(self.num_users, self.emb_size), mean=0, std=0.1)
        item_emb_weight = torch.nn.init.normal_(torch.empty(self.num_items, self.emb_size), mean=0, std=0.1)
        self.user_embeddings = torch.nn.Embedding(self.num_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = torch.nn.Embedding(self.num_items, self.emb_size, _weight=item_emb_weight)

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = []
        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            if perturbed:
                noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(noise, dim=1) * self.eps
            all_emb.append(ego_embeddings)
        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        return torch.split(all_emb, [self.num_users, self.num_items])

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        pos_score = (user_emb * pos_emb).sum(dim=1)
        neg_score = (user_emb * neg_emb).sum(dim=1)
        loss = -torch.log(1e-8 + torch.sigmoid(pos_score - neg_score)).mean()
        reg_loss = self.decay * (torch.norm(user_emb) + torch.norm(pos_emb))
        return loss, reg_loss

    def cl_loss(self, u_idx, i_idx, j_idx):
        u_idx = torch.tensor(u_idx)
        batch_users = torch.unique(u_idx).long().to(self.device)
        batch_pop, batch_unpop = utils.split_bacth_items(i_idx, self.pop_train)
        batch_pop = torch.unique(torch.tensor(batch_pop)).long().to(self.device)
        batch_unpop = torch.unique(torch.tensor(batch_unpop)).long().to(self.device)

        user_v1, item_v1 = self.forward(perturbed=True)
        user_v2, item_v2 = self.forward(perturbed=True)

        # User CL loss (normal)
        user_cl_loss = metrics.InfoNCE(user_v1[batch_users], user_v2[batch_users], self.temperature) * self.cl_rate

        # Item CL loss (Gradient reweighting based)
        pop_view1, pop_view2 = item_v1[batch_pop], item_v2[batch_pop]
        unpop_view1, unpop_view2 = item_v1[batch_unpop], item_v2[batch_unpop]

        pop_distance = 1 - F.cosine_similarity(pop_view1, pop_view2)
        unpop_distance = 1 - F.cosine_similarity(unpop_view1, unpop_view2)

        pop_weights = pop_distance / pop_distance.mean()
        unpop_weights = unpop_distance / unpop_distance.mean()

        pop_cl_loss = (self.gamma * (pop_weights * metrics.InfoNCE_i(pop_view1, pop_view2, item_v2[batch_unpop], self.temperature, self.lambda2))).mean()
        unpop_cl_loss = ((1 - self.gamma) * (unpop_weights * metrics.InfoNCE_i(unpop_view1, unpop_view2, item_v2[batch_pop], self.temperature, self.lambda2))).mean()

        item_cl_loss = (pop_cl_loss + unpop_cl_loss) * self.cl_rate
        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, user_cl_loss, item_cl_loss

    def batch_loss(self, u_idx, i_idx, j_idx):
        user_embedding, item_embedding = self.forward(perturbed=False)
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        neg_emb = item_embedding[j_idx]
        bpr_loss, l2_loss = self.bpr_loss(user_emb, pos_emb, neg_emb)
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx, j_idx)
        batch_loss = bpr_loss + l2_loss + cl_loss
        return batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss


def test(model):
    user_embedding, item_embedding = model.forward()
    return user_embedding.detach().cpu().numpy(), item_embedding.detach().cpu().numpy()


def train(config, data, model, optimizer, early_stopping, logger,train_step=1):
    model.train()
    for epoch in range(config.num_epoch):
        start = datetime.datetime.now()
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
            'cl_loss':0.0,
            'batch_loss': 0.0,
            'align_loss':0.0,
        }
        # train
        with tqdm(total=math.ceil(len(data.training_data) / config.batch_size), desc=f'Epoch {epoch}',
                  unit='batch')as pbar:
            for n, batch in enumerate(dataloader.next_batch_pairwise(data, config.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss= model.batch_loss(
                    user_idx, pos_idx, neg_idx)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_res['bpr_loss']+= bpr_loss.item()
                train_res['emb_loss'] += l2_loss.item()
                train_res['batch_loss'] += batch_loss.item()
                train_res['cl_loss']+= cl_loss.item()

                pbar.set_postfix({'loss (batch)': batch_loss.item()})
                pbar.update(1)
        train_res['bpr_loss'] = train_res['bpr_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['emb_loss'] = train_res['emb_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['batch_loss'] = train_res['batch_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['cl_loss'] = train_res['cl_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        

        user_emb, item_emb = model.forward()
        for _ in range(train_step):
            G1,G2=dataloader.user_items_2_group_pop(data)
            align_loss=utils.alignment_user(item_emb[G1],item_emb[G2])*config.align_reg
            optimizer.zero_grad()
            align_loss.backward()
            optimizer.step()
            train_res['align_loss'] += align_loss.item()
            


        train_res['align_loss'] = train_res['align_loss'] / train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        logger.info(training_logs)
        trin_time = datetime.datetime.now()

        # val
        model.eval()
        user_embedding, item_embedding = test(model)
        val_hr,val_recall, val_ndcg = mini_batch_test.test_acc_batch(
            data.val_U2I, data.train_U2I, user_embedding, item_embedding)
        logger.info('val_hr@100:{:.6f}   val_recall@100:{:.6f}   val_ndcg@100:{:.6f}   train_time:{}s   test_tiem:{}s'.format(
            val_hr, val_recall, val_ndcg, (trin_time-start).seconds, (datetime.datetime.now() - trin_time).seconds))

        # early_stopping
        early_stopping(val_hr, model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    



def main(config):

    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    file_name = '_'.join((str(config.layers),str(config.cl_rate),str(config.align_reg),str(config.gamma),str(config.lambda2), timestamp))

    result_path = '/'.join((config.result_path,
                           config.model, config.dataset_name, file_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    logger_file_name = os.path.join(result_path, 'train_logger')
    logger = utils.get_logger(logger_file_name)
    for name, value in vars(config).items():
        logger.info('%20s =======> %-20s' % (name, value))

    # Seed
    if config.seed:
        utils.setup_seed(config.seed)
    # config.device = 'cpu'


    # Load Data
    logger.info('------Load Data-----')
    data = dataloader.Data(config, logger)
    data.norm_adj = dataloader.LaplaceGraph(
        data.num_users, data.num_items, data.train_U2I).generate()
        



    # Load Model
    logger.info('------Load Model-----')
    model = PAAC(config, data)
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # EarlyStopping
    early_stopping = utils.EarlyStopping(logger,
        config.EarlyStop, verbose=True, path=result_path)

    # train model
    train(config, data, model, optimizer, early_stopping, logger)

    # #test model

    model_dict = model.load_state_dict(torch.load(result_path + "/best_val_epoch.pt"))
    user_embedding,item_embedding=test(model)

    val_hr,val_recall, val_ndcg = mini_batch_test.test_acc_batch(
        data.val_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\nval_hr@20:{:.6f}   val_recall@20:{:.6f}   val_ndcg@20:{:.6f} '.format(
        val_hr, val_recall, val_ndcg))
    test_OOD_hr, test_OOD_recall, test_OOD_ndcg = mini_batch_test.test_acc_batch(
        data.test_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\ntest_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f} '.format(
        test_OOD_hr, test_OOD_recall, test_OOD_ndcg))
    test_IID_hr, test_IID_recall, test_IID_ndcg = mini_batch_test.test_acc_batch(
        data.test_iid_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\ntest_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} '.format(
        test_IID_hr, test_IID_recall, test_IID_ndcg))
    return val_hr, val_recall, val_ndcg,test_OOD_hr, test_OOD_recall, test_OOD_ndcg ,test_IID_hr, test_IID_recall, test_IID_ndcg,result_path

if __name__ == '__main__':
    config=main_args()
    result_path = '/'.join((config.result_path,
                                config.model, config.dataset_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    f=open('/'.join((config.result_path,config.model, config.dataset_name))+'/best_performace.txt','a+')
    for cl_rate in ast.literal_eval(config.cl_rate_list):
        for layers in ast.literal_eval(config.layers_list):
            for align_reg in ast.literal_eval(config.align_reg_list):
                for temperature in ast.literal_eval(config.temperature_list):
                    for lambda2 in ast.literal_eval(config.lambada_list):
                        for gamma in ast.literal_eval(config.gama_list):
                            config.temperature=temperature
                            config.cl_rate=cl_rate
                            config.layers=layers
                            config.align_reg=align_reg
                            config.lambda2=lambda2
                            config.gamma=gamma
                            val_hr, val_recall, val_ndcg,test_OOD_hr, test_OOD_recall, test_OOD_ndcg ,test_IID_hr, test_IID_recall, test_IID_ndcg,result_path= main(config)
                            f.write('\n')
                            f.write('\n ====layers:{}===cl-rate:{}===align_reg:{}===gamma:{}====lambda2:{}\n  best_hr@20:{}=====best_recall@20:{}====best_ndcg@20:{}\n test_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f}\n test_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} \n  Resulst_path:{}\n '
                                    .format(config.layers,config.cl_rate,config.align_reg,config.gamma,config.lambda2,val_hr,val_recall,val_ndcg,test_OOD_hr, test_OOD_recall, test_OOD_ndcg ,test_IID_hr, test_IID_recall, test_IID_ndcg,result_path))
                            f.write('\n')
    f.close()
