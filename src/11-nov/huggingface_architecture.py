import torch
from transformers import PreTrainedModel
from transformers import PretrainedConfig


class Custom_Config(PretrainedConfig):
    model_type="mlp"

    def __init__(self, , **kwargs,):
        self.input_shape       = input_shape
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        
        super().__init__(**kwargs)
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

class Universal_Approximation_Architecture(torch.nn.Module):
    config_class = Custom_Config
    
    def __init__(self, input_shape, hidden_layer_size, output_layer_size):
        super().__init__(**kwargs)
        
        self.input_layer  = nn.Input (in_features = input_shape)
        self.hidden_layer = nn.Linear(neurons = hidden_layer_size, in_features = input_shape, activation="relu")
        self.output_layer = nn.Linear(neurons = output_layer_size, in_features = hidden_layer_size, activation=None)
        
        
    def forward(self, x_batch):
        x_batch = self.input_layer.transform(x_batch)
        x_batch = self.hidden_layer.transform(x_batch)
        y_preds_logits = self.output_layer.transform(x_batch)

        return y_preds_logits

class Transformer_Classifier(PreTrainedModel):
    config_class = Custom_Config
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        
        self.model = Basline_Architecture(config.input_shape, config.hidden_layer_size, config.output_layer_size)

    def forward(self, x_batch):
        return self.model(x_batch)

AutoConfig.register("resnet", Custom_Config)
AutoModel.register(Custom_Config, Transformer_Classifier)
AutoModelForImageClassification.register(Custom_Config, Transformer_Classifier)