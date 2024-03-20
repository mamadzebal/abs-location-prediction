import torch as T
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Model.RNN import RNN_Model

class Server(object):
    def __init__(self, w, LR, WEIGHT_DECAY_LAMBDA, NUM_ACTIONS, INPUT_SHAPE, FILE_NAME, CHECKPOINT_DIR):
        self.clients_update_w = []
        self.clients_loss = []
        self.model = RNN_Model(LEARNING_RATE=LR, WEIGHT_DECAY_LAMBDA=WEIGHT_DECAY_LAMBDA, ACTION_SHAPE=NUM_ACTIONS, INPUT_SHAPE=INPUT_SHAPE, FILE_NAME=FILE_NAME + "_q_server", SAVE_CHECKPOINT_DIRECTORY=CHECKPOINT_DIR)
        self.model.load_state_dict(w)
        
    def FedAvg(self):
        update_w_avg = copy.deepcopy(self.clients_update_w[0])
        for k in update_w_avg.keys():
            for i in range(1, len(self.clients_update_w)):
                update_w_avg[k] += self.clients_update_w[i][k]
            update_w_avg[k] = T.div(update_w_avg[k], len(self.clients_update_w))
            self.model.state_dict()[k] += update_w_avg[k]   
        return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

       

        
        
    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
