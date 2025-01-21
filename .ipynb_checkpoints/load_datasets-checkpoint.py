from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import PathologicalPartitioner, DirichletPartitioner
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

import numpy as np
from flwr_datasets.partitioner import Partitioner

from typing import Tuple

def load_datasets(args) -> Tuple[DataLoader, DataLoader, DataLoader]:

    def apply_transforms(batch):
        if args['DATASET'] == 'mnist':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
            batch["image"] = [transform(image) for image in batch["image"]]
        elif args['DATASET']=='cifar10':
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR 10
                ]
            )
            batch["img"] = [transform(img) for img in batch["img"]]
        else:
            print("ERROR! Dataset is not available")
        
        return batch
    
    print("-----------Hi-------------\n")

    if args['DATA_DISTRIBUTION'] == "IID":
        print("-----------IID-------------\n")

        # Set the partitioner to create 10 partitions
        partitioner = IidPartitioner(num_partitions=args['NUM_CLIENTS'])
        
        if args['DATASET'] == "mnist":
            fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": partitioner}, seed=args['SEED'])
            #fds = FederatedDataset(dataset="mnist", partitioners={"train": n_clients}, shuffle= False, seed=42)
        elif args['DATASET'] == "cifar10":
            fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner}, seed=args['SEED'])
            #fds = FederatedDataset(dataset="cifar10", partitioners={"train": n_clients}, shuffle= False, seed=42)
            #fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner}, seed=42)
        else:
            print("ERROR! Dataset non presente")

        # Create train/val for each partition and wrap it into DataLoader
        trainloaders = []
        valloaders = []
        
        for partition_id in range(args['NUM_CLIENTS']):
            partition = fds.load_partition(partition_id, "train")
            partition = partition.with_transform(apply_transforms)
            partition = partition.train_test_split(train_size=0.8, shuffle= False, seed=args['SEED'])
            trainloaders.append(DataLoader(partition["train"], batch_size=args['BATCH_SIZE'], shuffle=True))
            valloaders.append(DataLoader(partition["test"], batch_size=args['BATCH_SIZE']))
        testset = fds.load_split("test").with_transform(apply_transforms)
        #testloader = DataLoader(testset, batch_size=batch_size)
        testloader = DataLoader(testset, batch_size=args['BATCH_SIZE'] * args['NUM_CLIENTS'], shuffle=False)

        # verify IID distribution
        print(f"len trainloaders: {len(trainloaders)}")
        print(f"len valloaders: {len(valloaders)}")
        print(f"len testloader: {len(testloader)}")

        # for dataloader in trainloaders:
        #     for batch in dataloader:
        #         images, labels = batch["image"], batch["label"]
        #         print(labels[0])
        #         break
        
        #for batch in testloader:
        #    images, labels = batch["image"], batch["label"]
        #    print(labels[0])

        return trainloaders, valloaders, testloader
            
    
    elif args['DATA_DISTRIBUTION'] == "NON-IID":
        if  args['TYPE_DISTRIBUTION'] == 'dirichlet':

            partitioner = DirichletPartitioner(
                num_partitions=args['NUM_CLIENTS'],
                alpha=0.1,
                partition_by="label"
            )
        elif args['TYPE_DISTRIBUTION'] == 'pathological-ordered' or  args['TYPE_DISTRIBUTION'] == 'pathological-balanced': 
            
            partitioner = PathologicalPartitioner(
            num_partitions= args['NUM_CLIENTS'],
            partition_by="label",
            num_classes_per_partition=1,
            class_assignment_mode="first-deterministic"
            )
        else:
            print("ERROR! Type NON-IID distribution is not available")


        if args['DATASET'] == "mnist":
            fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": partitioner}, seed=args['SEED'])
        elif args['DATASET'] == "cifar10":
            fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner}, seed=args['SEED'])
        else:
            print("ERROR! Dataset non presente")


        if  args['TYPE_DISTRIBUTION'] == 'dirichlet':

            # Create train/val for each partition and wrap it into DataLoader
            trainloaders = []
            valloaders = []

            # for partition in partitions:
            #     partition = partition.with_transform(apply_transforms)
            for partition_id in range(args['NUM_CLIENTS']):
                partition = fds.load_partition(partition_id, "train")
                partition = partition.with_transform(apply_transforms)
                partition = partition.train_test_split(train_size=0.8, shuffle= False, seed=args['SEED'])
                trainloaders.append(DataLoader(partition["train"], batch_size=args['BATCH_SIZE'], shuffle=True))
                valloaders.append(DataLoader(partition["test"], batch_size=args['BATCH_SIZE']))
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=args['BATCH_SIZE'] * args['NUM_CLIENTS'], shuffle=False)

            return trainloaders, valloaders, testloader
        
        elif args['TYPE_DISTRIBUTION'] == 'pathological-ordered':
            partitions = []
            for partition_id in range(args['NUM_CLIENTS']):
                try:
                    partition = fds.load_partition(partition_id, "train")
                    partitions.append(partition)
                except IndexError:
                    print(f"Partition {partition_id} does not exist. Adjusting NUM_CLIENTS.")
                    args['NUM_CLIENTS'] = len(partitions)
                    break
        
            trainloaders = []
            valloaders = []
            for partition in partitions:
                partition = partition.with_transform(apply_transforms)
                partition = partition.train_test_split(train_size=0.8, shuffle=False, seed=args['SEED'])
                trainloaders.append(DataLoader(partition["train"], batch_size=args['BATCH_SIZE'], shuffle=True))
                valloaders.append(DataLoader(partition["test"], batch_size=args['BATCH_SIZE']))
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=args['BATCH_SIZE'] * args['NUM_CLIENTS'], shuffle=False)
        
            return trainloaders, valloaders, testloader

        
        elif args['TYPE_DISTRIBUTION'] == 'pathological-balanced':
            partitions = []
            for partition_id in range(args['NUM_CLIENTS']):
                try:
                    partition = fds.load_partition(partition_id, "train")
                    partitions.append(partition)
                except IndexError:
                    print(f"Partition {partition_id} does not exist. Adjusting NUM_CLIENTS.")
                    args['NUM_CLIENTS'] = len(partitions)
                    break
        
            trainloaders = []
            valloaders = []
            for partition in partitions:
                partition = partition.with_transform(apply_transforms)
                partition = partition.train_test_split(train_size=0.8, shuffle=False, seed=args['SEED'])
                trainloaders.append(DataLoader(partition["train"], batch_size=args['BATCH_SIZE'], shuffle=True))
                valloaders.append(DataLoader(partition["test"], batch_size=args['BATCH_SIZE']))
            testset = fds.load_split("test").with_transform(apply_transforms)
            testloader = DataLoader(testset, batch_size=args['BATCH_SIZE'] * args['NUM_CLIENTS'], shuffle=False)
        
            return trainloaders, valloaders, testloader

        else:
            print("ERROR! Type NON-IID distribution is not available")


