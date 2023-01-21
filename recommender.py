import torch
import random
import config


class Recommender:
    def __init__(self, model, feedbacks_by_users, items_names, top_k, sampling_amount):
        self.model = model
        self.feedbacks_by_users = feedbacks_by_users
        self.items_names = items_names
        self.top_k = top_k
        self.sampling_amount = sampling_amount

    def get_recommendations(self, user_id):
        # sampling random items from the items that the user is not interacted with
        sampled_items = random.sample(self.feedbacks_by_users['negative_feedbacks'][user_id], self.sampling_amount)

        # preparing data input for prediction
        users = []
        for i in range(self.sampling_amount):
            users.append(user_id)  # same user for each of item prediction
        prediction_input = PredictionInput(users, sampled_items, config.device)
        user, items = PredictionInputLoader(prediction_input, self.sampling_amount).get_prediction_batch_input()

        # get prediction
        self.model.to(config.device)
        predictions = self.model(user, items)

        # get top K recommendations
        ratings, indexes = torch.topk(predictions, self.top_k)
        recommendations_items_id = torch.take(items, indexes).cpu().numpy().tolist()
        recommendations_items_names = self.get_items_names_by_id(recommendations_items_id)
        return recommendations_items_id, recommendations_items_names

    def get_items_names_by_id(self, items_id):
        items_names = []
        for i in range(len(items_id)):
            items_names.append(self.items_names[items_id[i]])
        return items_names

    def get_users_amount(self):
        return len(self.feedbacks_by_users)

    def get_items_amount(self):
        return len(self.items_names)


class PredictionInput(torch.utils.data.Dataset):
    def __init__(self, users, items, device):
        super(PredictionInput, self).__init__()
        self.users = users
        self.items = items
        self.device = device

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]

        return (
            torch.tensor(user, dtype=torch.long).to(self.device),
            torch.tensor(item, dtype=torch.long).to(self.device)
        )


class PredictionInputLoader:
    def __init__(self, prediction_input, sampling_amount):
        self.data_loader = torch.utils.data.DataLoader(prediction_input, batch_size=sampling_amount, shuffle=False)

    def get_prediction_batch_input(self):
        return next(iter(self.data_loader))  # this data loader consist of only one batch, so it's return the first batch instead of iterating on data loader


def main():
    print("Loading the recommender ... ")
    recommender = torch.load(config.recommender_path, map_location=config.device)

    print("Enter user ID (from 0 to ", recommender.get_users_amount(), "): ")
    user_id = int(input())
    while user_id >= recommender.get_users_amount() or user_id < 0:
        print("user id not exist")
        user_id = int(input("Enter valid user ID (from 0 to " + recommender.get_users_amount() + "): "))

    print("Getting model recommendations for user ", user_id, " ... ")
    recommendations_ids, recommendations_names = recommender.get_recommendations(user_id)
    print("recommended items id's: ")
    print(recommendations_ids)
    print("recommended items names: ")
    for i in range(len(recommendations_names)):
        print(i+1, ". ", recommendations_names[i])


if __name__ == "__main__":
    main()
