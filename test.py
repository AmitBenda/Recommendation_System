import math
import torch


class Test:
    def __init__(self, model, test_dataset, top_k):
        self.hit_ratio = []
        self.NDCG = []

        for user, items, _ in test_dataset:  # iterating over a batch for each user
            # each batch is consist of the same user with a different items to test the recommendations for the user
            predictions = model(user, items)

            # getting top K recommended items id's
            ratings, indexes = torch.topk(predictions, top_k)
            recommendations = torch.take(items, indexes).cpu().numpy().tolist()

            # calculate hit ratio and NDCG
            positive_item = items[0].item()
            self.hit_ratio.append(self.calculate_hit_ratio(positive_item, recommendations))
            self.NDCG.append(self.calculate_NDCG(positive_item, recommendations))

    @staticmethod
    def calculate_hit_ratio(positive_item, recommendations):
        if positive_item in recommendations:
            return 1
        return 0

    @staticmethod
    def calculate_NDCG(positive_item, recommendations):
        if positive_item in recommendations:
            index = recommendations.index(positive_item)
            log2_index = math.log2(index + 2)  # '+2' to avoid log[0] (undefined) and divide by log[1]=0 (undefined), and starting from 2 is the best test scenario that NDCG will return 1 (the larger the index, the lower the NDCG will be)
            return math.pow(log2_index, -1)
        return 0

    def get_hit_ratio_average(self):
        return sum(self.hit_ratio) / len(self.hit_ratio)

    def get_NDCG_average(self):
        return sum(self.NDCG) / len(self.NDCG)


