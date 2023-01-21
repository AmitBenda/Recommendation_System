from random import random
import pandas as pd
import torch
import random


class RatingsDataLoader:
    def __init__(self, config):
        self.users_list = []
        self.items_list = []
        self.items_names = []
        self.feedbacks_by_users = []
        self.config = config
        dataset_df, items_df = self.read_dataset()
        self.train_dataset, self.test_dataset = self.pre_process_dataset(dataset_df, items_df)

    def get_train_dataset(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def get_test_dataset(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.test_negative_feedback_ratio + 1, shuffle=False)

    def get_users_amount(self):
        return len(self.users_list)

    def get_items_amount(self):
        return len(self.items_list)

    def get_items_names(self):
        return self.items_names

    def get_feedbacks_by_users(self):
        return self.feedbacks_by_users

    def pre_process_dataset(self, dataset_df, items_df):
        print("pre-processing dataset ... ")
        dataset_df = self.filter_dataset_by_minimum_time(dataset_df, items_df)
        train_df, test_df = self.split_positive_to_train_test(dataset_df)
        train_dataset, test_dataset = self.get_train_test_datasets_from_df(dataset_df, train_df, test_df)
        return train_dataset, test_dataset

    def filter_dataset_by_minimum_time(self, dataset_df, items_df):
        print("filtering dataset ... ")

        # filter by timestamp
        dataset_df = dataset_df[dataset_df['timestamp'] > self.config.minimum_timestamp_filter].reset_index(drop=True)
        items_list = dataset_df['movieId'].unique()
        items_df = items_df[items_df['movieId'].isin(items_list)].reset_index(drop=True)

        # adjusting the new id's to users and items after the filtering
        dataset_df, items_df = self.remap_users_items_id(dataset_df, items_df)

        # filter by removing the users that have less than 20 feedbacks
        feedbacks_by_users = (dataset_df.groupby('userId')['movieId'].apply(set).reset_index().rename(columns={'movieId': 'positive_feedbacks'}))
        for i in range(len(feedbacks_by_users)):
            if len(feedbacks_by_users['positive_feedbacks'][i]) < self.config.minimum_user_feedbacks_amount:
                dataset_df = dataset_df[dataset_df['userId'] != i].reset_index(drop=True)
        items_list = dataset_df['movieId'].unique()
        items_df = items_df[items_df['movieId'].isin(items_list)].reset_index(drop=True)

        # adjusting the new id's to users and items after the filtering
        dataset_df, items_df = self.remap_users_items_id(dataset_df, items_df)

        sorted_items_by_id = items_df.sort_values('movieId')
        self.items_names = sorted_items_by_id['title'].unique()
        self.users_list = dataset_df['userId'].unique()
        self.items_list = dataset_df['movieId'].unique()
        print("Amount of positive feedbacks: ", len(dataset_df))
        print("Amount of users: ", len(self.users_list))
        print("Amount of items: ", len(self.items_list))
        return dataset_df

    @staticmethod
    def remap_users_items_id(dataset_df, items_df):
        users_list = dataset_df['userId'].unique()
        items_list = dataset_df['movieId'].unique()
        users_map_id = {old_id: index for index, old_id in enumerate(users_list)}
        items_map_id = {old_id: index for index, old_id in enumerate(items_list)}
        dataset_df['userId'] = dataset_df['userId'].apply(lambda old_id: users_map_id[old_id])
        dataset_df['movieId'] = dataset_df['movieId'].apply(lambda old_id: items_map_id[old_id])
        items_df['movieId'] = items_df['movieId'].apply(lambda old_id: items_map_id[old_id])
        return dataset_df, items_df

    @staticmethod
    def split_positive_to_train_test(dataset_df):
        print("split dataset into test & train... ")

        # split by 'LOO' (leave one out) protocol
        dataset_df['time_order'] = dataset_df.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)  # for each user, ranking the feedback by time order, from the latest to the oldest
        test = dataset_df.loc[dataset_df['time_order'] == 1].reset_index()  # the latest feedback for each user go to test
        train = dataset_df.loc[dataset_df['time_order'] > 1].reset_index()  # the rest of the feedbacks goes to train
        return train[['userId', 'movieId']], test[['userId', 'movieId']]

    def get_train_test_datasets_from_df(self, dataset_df, train_df, test_df):
        # for each user, define a list of all positive feedbacks and a list of all negative feedbacks
        items_set = set(dataset_df['movieId'].unique())
        self.feedbacks_by_users = (dataset_df.groupby('userId')['movieId'].apply(set).reset_index().rename(columns={'movieId': 'positive_feedbacks'}))
        self.feedbacks_by_users['negative_feedbacks'] = self.feedbacks_by_users['positive_feedbacks'].apply(lambda user_positive_feedbacks: items_set - user_positive_feedbacks)

        print("getting negative samples ... ")
        train_negative_samples = []
        test_negative_samples = []
        for i in range(len(self.feedbacks_by_users)):
            negative_sample_amount = len(self.feedbacks_by_users['positive_feedbacks'][i]) * self.config.train_negative_feedback_ratio
            if negative_sample_amount > len(self.feedbacks_by_users['negative_feedbacks'][i]):
                negative_sample_amount = len(self.feedbacks_by_users['negative_feedbacks'][i])
            train_negative_samples.append(random.sample(self.feedbacks_by_users['negative_feedbacks'][i], negative_sample_amount))
            test_negative_samples.append(random.sample(self.feedbacks_by_users['negative_feedbacks'][i], self.config.test_negative_feedback_ratio))

        print("getting test_dataset ... ")
        test_dataset = self.get_dataset_from_df(test_df, self.config.test_negative_feedback_ratio, test_negative_samples)

        print("getting train_dataset ... ")
        train_dataset = self.get_dataset_from_df(train_df, self.config.train_negative_feedback_ratio, train_negative_samples)

        return train_dataset, test_dataset

    def get_dataset_from_df(self, dataset_df, negative_feedback_ratio, negative_samples):
        users = []
        items = []
        ratings = []
        for row in dataset_df.itertuples():
            users.append(row.userId)
            items.append(row.movieId)
            ratings.append(1.0)  # positive feedback (from explicit to implicit feedback)
            for i in range(negative_feedback_ratio):  # add the negative feedbacks
                if len(negative_samples[row.userId]) > 0:
                    users.append(row.userId)
                    items.append(negative_samples[row.userId].pop())
                    ratings.append(0.0)  # negative feedback (from explicit to implicit feedback)
        return RatingsDataset(users, items, ratings, self.config.device)

    def read_dataset(self):
        print("reading dataset ... ")
        path = self.config.local_dataset_path
        dataset_df = pd.read_csv(path)
        path = self.config.local_items_names_path
        items_df = pd.read_csv(path)
        return dataset_df, items_df


class RatingsDataset(torch.utils.data.Dataset):
    def __init__(self, users, items, ratings, device):
        super(RatingsDataset, self).__init__()
        self.users = users
        self.items = items
        self.ratings = ratings
        self.device = device

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]
        rating = self.ratings[index]

        return (
            torch.tensor(user, dtype=torch.long).to(self.device),
            torch.tensor(item, dtype=torch.long).to(self.device),
            torch.tensor(rating, dtype=torch.float).to(self.device)
        )

