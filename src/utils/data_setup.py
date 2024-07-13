"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from .common import *

NUM_WORKERS = os.cpu_count()

# Normalization values for the different datasets
NORMALIZE_DICT = {
    'mnist': dict(mean=(0.1307,), std=(0.3081,)),
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'animaux': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'breast': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'histo': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'MRI': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'DNA': None
    }

def read_and_prepare_data(file_path, size=6, model_name='all-MiniLM-L6-v2'):
    """
    Reads DNA sequence data from a text file and prepares it for modeling.
    """
    # Read data from file
    data = pd.read_table(file_path)

    # Function to extract k-mers from a sequence
    def getKmers(sequence):
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

    # Function to preprocess data
    def preprocess_data(data):
        data['words'] = data['sequence'].apply(lambda x: getKmers(x))
        data = data.drop('sequence', axis=1)
        return data

    # Preprocess data
    data = preprocess_data(data)

    def kmer_lists_to_texts(kmer_lists):
        return [' '.join(map(str, l)) for l in kmer_lists]

    data['texts'] = kmer_lists_to_texts(data['words'])

    # Prepare data for modeling
    texts = data['texts'].tolist()
    y_data = data.iloc[:, 0].values
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    del model
    return embeddings, y_data


def split_data_client(dataset, num_clients, seed):
    """
    This function is used to split the dataset into train and test for each client.
    :param dataset: the dataset to split (type: torch.utils.data.Dataset)
    :param num_clients: the number of clients
    :param seed: the seed for the random split
    """
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds


# Define model, architecture and dataset
# The DataLoaders downloads the training and test data that are then normalized.
def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/", data_path_val=""):
    """
    This function is used to load the dataset and split it into train and test for each client.
    :param num_clients: the number of clients
    :param batch_size: the batch size
    :param resize: the size of the image after resizing (if None, no resizing)
    :param seed: the seed for the random split
    :param num_workers: the number of workers
    :param splitter: percentage of the training data to use for validation. Example: 10 means 10% of the training data
    :param dataset: the name of the dataset in the data folder
    :param data_path: the path of the data folder
    :param data_path_val: the absolute path of the validation data (if None, no validation data)
    :return: the train and test data loaders
    """
    
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])] if dataset!="DNA" else None
    print(dataset)

    if dataset == "cifar":
        # Download and transform CIFAR-10 (train and test)
        transformer = transforms.Compose(
            list_transforms
        )
        trainset = datasets.CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)
    
    elif dataset == "DNA":

        human_X, human_y = read_and_prepare_data(data_path + dataset + '/human.txt')
        X_train, X_test, y_train, y_test = train_test_split(human_X, human_y, test_size=0.2, random_state=seed)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        trainset = TensorDataset(X_train, y_train)
        testset = TensorDataset(X_test, y_test)        
        
    else:
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset)
        supp_ds_store(data_path + dataset + "/Training")
        supp_ds_store(data_path + dataset + "/Testing")
        trainset = datasets.ImageFolder(data_path + dataset + "/Training", transform=transformer)
        testset = datasets.ImageFolder(data_path + dataset + "/Testing", transform=transformer)

    if dataset != "DNA":
        print(f"The training set is created for the classes : {trainset.classes}")
    else:
        print("The training set is created for the classes: ('0', '1', '2', '3', '4', '5', '6')")

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets_train = split_data_client(trainset, num_clients, seed)
    if data_path_val and dataset != "DNA":
        valset = datasets.ImageFolder(data_path_val, transform=transformer)
        datasets_val = split_data_client(valset, num_clients, seed)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        if data_path_val:
            # if we already have a validation dataset
            trainloaders.append(DataLoader(datasets_train[i], batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(datasets_val[i], batch_size=batch_size))

        else:
            len_val = int(len(datasets_train[i]) * splitter / 100)  # splitter % validation set
            len_train = len(datasets_train[i]) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(datasets_train[i], lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
