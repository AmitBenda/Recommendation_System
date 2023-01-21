import torch
import config


class NCF(torch.nn.Module):
    def __init__(self, config, amount_of_users, amount_of_items):
        super(NCF, self).__init__()
        # model configurations
        self.amount_of_users = amount_of_users
        self.amount_of_items = amount_of_items
        self.mlp_users_latent_vector_size = config.mlp_users_latent_vector_size
        self.mlp_items_latent_vector_size = config.mlp_items_latent_vector_size
        self.gmf_latent_vector_size = config.gmf_latent_vector_size
        self.mlp_layers_sizes = config.mlp_layers_sizes

        # model embeddings
        self.mlp_users_embedding = torch.nn.Embedding(num_embeddings=self.amount_of_users, embedding_dim=self.mlp_users_latent_vector_size)
        self.mlp_items_embedding = torch.nn.Embedding(num_embeddings=self.amount_of_items, embedding_dim=self.mlp_items_latent_vector_size)
        self.gmf_users_embedding = torch.nn.Embedding(num_embeddings=self.amount_of_users, embedding_dim=self.gmf_latent_vector_size)
        self.gmf_items_embedding = torch.nn.Embedding(num_embeddings=self.amount_of_items, embedding_dim=self.gmf_latent_vector_size)

        # model MLP
        self.mlp_layers = torch.nn.ModuleList()
        self.mlp_layers.append(torch.nn.Linear(self.mlp_users_latent_vector_size + self.mlp_items_latent_vector_size, self.mlp_layers_sizes[0]))
        for i in range(len(self.mlp_layers_sizes) - 1):
            self.mlp_layers.append(torch.nn.Linear(self.mlp_layers_sizes[i], self.mlp_layers_sizes[i + 1]))
            self.mlp_layers.append(torch.nn.ReLU())

        # model output layer
        self.output_layer = torch.nn.Linear(in_features=self.gmf_latent_vector_size + self.mlp_layers_sizes[-1], out_features=1)
        self.output_normalizer = torch.nn.Sigmoid()

    def forward(self, user_input_vector, item_input_vector):
        # calculate embeddings
        mlp_users_latent_vector = self.mlp_users_embedding(user_input_vector)
        mlp_items_latent_vector = self.mlp_items_embedding(item_input_vector)
        gmf_users_latent_vector = self.gmf_users_embedding(user_input_vector)
        gmf_items_latent_vector = self.gmf_items_embedding(item_input_vector)

        # calculate MLP
        mlp_vector = torch.cat([mlp_users_latent_vector, mlp_items_latent_vector], dim=-1)  # dim=-1 in order to perform the concatenation by the last dimension, in this case by vector values => a vector concatenation
        for i in range(len(self.mlp_layers)):
            mlp_vector = self.mlp_layers[i](mlp_vector)

        # calculate GMF
        gmf_vector = torch.mul(gmf_users_latent_vector, gmf_items_latent_vector)

        # calculate output layer
        last_layer_vector = torch.cat([mlp_vector, gmf_vector], dim=-1)
        output = self.output_layer(last_layer_vector)
        prediction = self.output_normalizer(output).squeeze()
        return prediction


def main():
    print("Loading model ... ")
    model = torch.load(config.model_path, map_location=config.device)

    print("Enter user ID (from 0 to ", model.amount_of_users, "): ")
    user_id = int(input())
    while user_id >= model.amount_of_users or user_id < 0:
        print("user id not exist")
        user_id = int(input("Enter valid user ID (from 0 to " + model.amount_of_users + "): "))

    print("Enter item ID (from 0 to ", model.amount_of_items, "): ")
    item_id = int(input())
    while item_id >= model.amount_of_items or item_id < 0:
        print("item id not exist")
        item_id = int(input("Enter valid item ID (from 0 to " + model.amount_of_items + "): "))

    print("Getting model prediction for user ", user_id, " with item ", item_id, " ...")
    user = torch.tensor(user_id, dtype=torch.long).to(config.device)
    item = torch.tensor(item_id, dtype=torch.long).to(config.device)
    prediction = model(user, item).item()
    print("model prediction: ")
    print(prediction)


if __name__ == "__main__":
    main()
