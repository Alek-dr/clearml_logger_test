from clearml import Task

from config import TrainingConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_shape):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def train_pytorch_model(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]

    trainset = dataset(X_train, y_train)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)

    model = NeuralNetworkClassificationModel(input_dim)
    learning_rate = 0.01
    epochs = 700
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    loss = 0
    acc = 0
    for i in range(epochs):
        for j, (x_train, y_train) in enumerate(trainloader):
            # calculate output
            output = model(x_train)

            # calculate loss
            loss = loss_fn(output, y_train.reshape(-1, 1))

            # accuracy
            predicted = model(torch.tensor(x_train, dtype=torch.float32))
            acc = (
                predicted.reshape(-1).detach().numpy().round() == y_train.numpy()
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % 50 == 0:
            print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))
    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(X_test, dtype=torch.float32))
        test_acc = (predicted.reshape(-1).detach().numpy().round() == y_test).mean()
        print("Test accuracy : {}".format(test_acc))

    example_forward_input = torch.rand(1, 2)
    module = torch.jit.trace(model, example_forward_input)
    torch.jit.save(module, "models/torch/1/model.pt")

def train_network(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs,
    train_losses,
    test_losses,
):
    for epoch in range(num_epochs):
        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # forward feed
        output_train = model(X_train)

        # calculate the loss
        loss_train = criterion(output_train, y_train)

        # backward propagation: calculate gradients
        loss_train.backward()

        # update the weights
        optimizer.step()

        output_test = model(X_test)
        loss_test = criterion(output_test, y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}"
            )


def main():
    config = TrainingConfig.parse_raw("configs/train_config.yaml")

    # connect packages
    Task.add_requirements('./requirements.txt')

    task = Task.init(
        project_name=config.clearml_project,
        task_name=config.clearml_task_name,
        output_uri=True,
        auto_connect_arg_parser=False,
        auto_connect_frameworks={'pytorch': False}
        # We disconnect pytorch auto-detection, because we added manual model save points in the code
    )

    task.set_base_docker(
        docker_image="pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
        docker_setup_bash_script=[
            "git config --global http.sslverify false",
            "apt-get update",
            "apt-get install ffmpeg libsm6 libxext6  -y",
            "apt-get install gcc -y",
            "export PYTHONPATH='${PYTHONPATH}':pwd"
        ]
    )
    task.connect(config, 'Args')
    task_id = task.get_parameter("Args/task_id")

    if task_id is not None:
        dataset_id = Task.get_task(task_id).get_parameters()["Args/dataset_id"]
        dataset_path = f"clearml://{dataset_id}"
        config.data = dataset_path


    train_pytorch_model(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
