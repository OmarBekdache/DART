import argparse
import os
import sys
from dataclasses import dataclass

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.init_args()
        self.parse_args()
        self.makedir()
        self.print_and_save_args()
        
    #Add arguments to the parser
    def init_args(self):
        #Run Arguments
        self.parser.add_argument('--run_name', type=str, required=True, help='run name')
        self.parser.add_argument('--gpu_idx', type=int, required=True, help="GPU IDs")
        self.parser.add_argument('--seed', type=int, required=True, help='seed')
        
        #Federated Learning Setup Arguments 
        self.parser.add_argument('--method', type=str, required=True, help='FL method to use')
        self.parser.add_argument('--model_type', type=str, required=True, help='Model executed on clients') #See models folder for list of available models
        self.parser.add_argument('--client_ds_name', type=str, required=True, help='Client dataset name')
        self.parser.add_argument('--server_ds_name', type=str, required=True, help='Server dataset name')
        self.parser.add_argument('--non_iid_dirichlet_alpha', type=float, required=True, help='Dirichlet distribution alpha parameter for non-iid data distribution among clients')

        #FL General Hyperparameters
        self.parser.add_argument('--num_clients', type=int, required=True, help='Number of clients') #Denoted by K
        self.parser.add_argument('--local_epochs', type=int, required=True ,help='Number of local epochs before aggregation') #Denoted by E
        self.parser.add_argument('--clients_per_round', type=int, required=True ,help='Number of clients selected per round') #Denoted by m
        self.parser.add_argument('--total_communication_rounds', type=int, required=True ,help='Number of total communication rounds') #Denoted by T
        
        #Client Hyperparameters
        self.parser.add_argument('--client_batch_size', type=int, required=True ,help='Client local batch size') #Denoted by B
        self.parser.add_argument('--client_learning_rate', type=float, required=True ,help='Client learning rate') #Denoted by eta_c
        self.parser.add_argument('--client_num_workers', type=int, required=True ,help='Client num workers') 

        #Server Hyperparameters
        self.parser.add_argument('--server_learning_rate', type=float, required=True ,help='Server learning rate') #Denoted by eta_s (if applicable)
        self.parser.add_argument('--server_batch_size', type=int, required=True ,help='Server batch size') #Denoted by B_s (if applicable)
        self.parser.add_argument('--server_data_ratio', type=float, required=True ,help='If the server has same distribution as client, this determines the fraction of data on the server')
        self.parser.add_argument('--server_val_data_ratio', type=float, required=True ,help='If the server has same distribution as client, this determines the fraction of train data vs val data on the server')
        self.parser.add_argument('--server_num_workers', type=int, required=True ,help='Server num workers')
        
        #AugMix Arguments
        self.parser.add_argument('--mixture_width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
        self.parser.add_argument('--mixture_depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
        self.parser.add_argument('--aug_severity', default=3, type=int, help='Severity of base augmentation operators')
        self.parser.add_argument('--no_jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
        self.parser.add_argument('--all_ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')
        
        #DART Hyperparameters
        self.parser.add_argument('--DART_alpha', type=float, required=True)
        self.parser.add_argument('--max_DART_epochs', type=int, required=True)
        self.parser.add_argument('--DART_patience', type=int, required=True)
        
        self.parser.add_argument("--DART_only", action="store_true")
        self.parser.add_argument('--checkpoint_path', type=str)
        self.parser.add_argument('--server_method', type=str, required=True)
        
        self.parser.add_argument('--progress_bar', action="store_true")
        
        self.parser.add_argument('--mu_fedprox', type=float)
        
        self.parser.add_argument('--fed_adam_server_lr', type=float)
        self.parser.add_argument('--beta1', type=float)
        self.parser.add_argument('--beta2', type=float)
        
    #Read and store arguments
    def parse_args(self):
        args = self.parser.parse_args()
        
        #Run Arguments
        self.run_name = args.run_name
        self.gpu_idx = args.gpu_idx
        self.seed = args.seed
        self.DART_only = args.DART_only
        self.checkpoint_path = args.checkpoint_path
        
        #Federated Learning Setup Arguments 
        self.method = args.method
        self.model_type = args.model_type
        self.client_ds_name = args.client_ds_name
        self.server_ds_name = args.server_ds_name
        self.non_iid_dirichlet_alpha = args.non_iid_dirichlet_alpha

        #FL General Hyperparameters
        self.num_clients = args.num_clients
        self.local_epochs = args.local_epochs
        self.clients_per_round = args.clients_per_round
        self.total_communication_rounds = args.total_communication_rounds
        
        #Client Hyperparameters
        self.client_batch_size = args.client_batch_size
        self.client_learning_rate = args.client_learning_rate
        self.client_num_workers = args.client_num_workers

        #Server Hyperparameters
        self.server_learning_rate = args.server_learning_rate
        self.server_batch_size = args.server_batch_size
        self.server_data_ratio = args.server_data_ratio
        self.server_val_data_ratio = args.server_val_data_ratio
        self.server_num_workers = args.server_num_workers
        
        #Augmix Arguments
        self.mixture_width = args.mixture_width
        self.mixture_depth = args.mixture_depth
        self.aug_severity = args.aug_severity
        self.no_jsd = args.no_jsd
        self.all_ops = args.all_ops
        
        #DART Hyperparameters
        self.DART_alpha = args.DART_alpha
        self.max_DART_epochs = args.max_DART_epochs
        self.DART_patience = args.DART_patience
        
        self.server_method = args.server_method
        
        self.progress_bar = args.progress_bar
        
        self.mu_fedprox = args.mu_fedprox
        
        self.fed_adam_server_lr = args.fed_adam_server_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        
    #Make directory for run files
    def makedir(self):
        self.directory = self.run_name
        
        if os.path.exists(self.directory):
            print(f"Error: Directory '{self.directory}' already exists. Please change run name.")
            sys.exit(1) 
        else:
            print(f"Directory created successfully. Please find run data in '{self.directory}'. \n")

        os.makedirs(self.directory)
        
    #Print and save arguments
    def print_and_save_args(self):
        file_path = os.path.join(self.directory, "config.txt")
        
        with open(file_path, 'w') as file:
            line = "Run Arguments"
            print(line)
            file.write(line+"\n")
            for key, value in vars(self).items():
                if not key.startswith('_') and key != 'parser':  
                    line = f"{key}: {value}"
                    print(line)
                    file.write(line+"\n")
        print()

    def get_dataset_config(self):
        dataset_config = DatasetConfig(
            seed = self.seed,
            num_clients=self.num_clients,
            method=self.method,
            client_ds_name=self.client_ds_name,
            client_batch_size=self.client_batch_size,
            client_num_workers=self.client_num_workers,
            server_ds_name=self.server_ds_name,
            server_batch_size=self.server_batch_size,
            server_num_workers=self.server_num_workers,
            server_data_ratio=self.server_data_ratio,
            non_iid_dirichlet_alpha=self.non_iid_dirichlet_alpha
        )
        print(f"Dataset Config: {dataset_config}")
        return dataset_config

    
        
@dataclass
class DatasetConfig:
    seed: int
    num_clients: int
    method: str

    client_ds_name: str
    client_batch_size: int
    client_num_workers: int

    server_ds_name: str
    server_batch_size: int
    server_num_workers: int
    server_data_ratio: float

    non_iid_dirichlet_alpha: float