import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using model on ", device)

# dataset config
local_dataset_path = "./dataset/ratings.csv"
local_items_names_path = "./dataset/movies.csv"
minimum_timestamp_filter = 1546300800  # 0/0/2019
train_negative_feedback_ratio = 4
test_negative_feedback_ratio = 100
minimum_user_feedbacks_amount = 20

# model config
mlp_users_latent_vector_size = 32
mlp_items_latent_vector_size = 32
gmf_latent_vector_size = 32
mlp_layers_sizes = [32, 16, 8]  # first layer is 64 from concatenating the two embedding vectors in size 32

# train & test config
lr = 0.0001
epochs = 5
batch_size = 256
model_path = "./model/model.pt"
recommender_path = "./model/recommender.pt"
top_k = 10

# GUI config
gui_top_k = 10
gui_sampling_amount = 100
