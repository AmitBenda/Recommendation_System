import torch
import config
from model import NCF
from test import Test
from data import RatingsDataLoader
from recommender import Recommender
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.lr = config.lr
        self.epochs = config.epochs
        self.device = config.device
        self.dataset = data_loader.get_train_dataset()
        self.test_dataset = data_loader.get_test_dataset()
        self.model = NCF(config, data_loader.get_users_amount(), data_loader.get_items_amount())
        self.model.to(self.device)
        self.loss_function = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.test_plots = SummaryWriter()

    def train(self):
        print("Dataset len: ", len(self.dataset) * self.config.batch_size)
        print("Batch size: ", self.config.batch_size)
        for epoch in range(self.epochs):
            i = 1
            epoch_loss = 0
            print("Epoch: ", epoch + 1, "/", self.epochs, ", Iteration: ", i, "/", len(self.dataset))
            for users, items, feedbacks in self.dataset:
                # Iterate over batch of users and items
                if i % 1000 == 0:
                    print("Epoch: ", epoch+1, "/", self.epochs, ", Iteration: ", i, "/", len(self.dataset))
                i += 1
                self.optimizer.zero_grad()  # zeros the gradient
                prediction = self.model(users, items)
                loss = self.loss_function(prediction, feedbacks)
                epoch_loss += loss.item()
                loss.backward()  # calculate backward
                self.optimizer.step()  # adjusting weights

            print("Epoch: ", epoch + 1, "/", self.epochs, ", Iteration: ", i-1, "/", len(self.dataset))
            print("Loss: ", epoch_loss / len(self.dataset))
            test = Test(self.model, self.test_dataset, self.config.top_k)
            print("Average Hit ratio", test.get_hit_ratio_average())
            print("Average NDCG", test.get_NDCG_average())
            self.test_plots.add_scalar('TrainLoss', epoch_loss / len(self.dataset), epoch+1)
            self.test_plots.add_scalar('HR@10_Average', test.get_hit_ratio_average(), epoch+1)
            self.test_plots.add_scalar('NDCG@10_Average', test.get_NDCG_average(), epoch+1)

        self.test_plots.close()
        print("saving model ...")
        recommender = Recommender(self.model, self.data_loader.get_feedbacks_by_users(), self.data_loader.get_items_names(), self.config.gui_top_k, self.config.gui_sampling_amount)
        torch.save(recommender, self.config.recommender_path)
        torch.save(self.model, self.config.model_path)
        print("model saved !")


def main():
    data_loader = RatingsDataLoader(config)
    print("Starting training ...")
    Train(config, data_loader).train()


if __name__ == "__main__":
    main()

