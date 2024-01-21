class LoadSaveFile:
    def __init__(
        self,
        dataset_name,
        batch_size,
        optimizer_name,
        logs_name,
        split_ratio=0,
        num_layers_level=None,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.logs_name = logs_name
        self.split_ratio = split_ratio
        self.num_layers_level = num_layers_level


    def get_file_path(self):
        if self.dataset_name == "ImageNet2012":
            if self.num_layers_level != None:
                raise ValueError(
                    "num_layers_level must be None when dataset_name is ImageNet2012"
                )
            self.file_name = f"MyResNet34_{self.batch_size}_{self.optimizer_name}"
            self.dir_name = f"MyResNet34_{self.dataset_name}_{self.batch_size}_{self.optimizer_name}"
        else:
            self.file_name = f"MyResNet{self.num_layers_level*6+2}_{self.batch_size}_{self.optimizer_name}"
            self.dir_name = f"MyResNet{self.num_layers_level*6+2}_{self.dataset_name}_{self.batch_size}_{self.optimizer_name}"

        if self.split_ratio != 0:
            self.file_name += f"_{int(self.split_ratio*100)}"
            self.dir_name += f"_{int(self.split_ratio*100)}"
        return self.dir_name, self.file_name
    
    def get_dataloader(self):
        
    def get_model(self):
        
        return 
    