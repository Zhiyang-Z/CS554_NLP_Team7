import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoTokenizer
from model.gpt import GPT
import yaml

# configure
with open("pretrain_config.yaml", "r") as f:
    config = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

class MyModelConfig(PretrainedConfig):
    model_type = "my_gpt_pretrain"

    def __init__(self,
                v_size = tokenizer.vocab_size,
                train_length = config['pretraining']['pretrain_length'],
                n_dim = config['model']['n_dim'],
                n_layer = config['model']['n_layer'],
                n_head = config['model']['n_head'],
                dim_head = config['model']['dim_head'],
                ff_ratio = config['model']['ff_ratio'],
                dropout_rate = config['model']['dropout_rate'],
                device = f'cpu',
                ex_ratio = config['model']['ex_ratio'],
                **kwargs):
        self.vocab_size = v_size
        self.train_length = train_length
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.dim_head = dim_head
        self.ff_ratio = ff_ratio
        self.dropout_rate = dropout_rate
        self.device = device
        self.ex_ratio = ex_ratio
        super().__init__(**kwargs)

class MyModelForCausalLM(PreTrainedModel):
    config_class = MyModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(v_size = config.vocab_size,
                         train_length = config.train_length,
                         n_dim = config.n_dim,
                         n_layer = config.n_layer,
                         n_head = config.n_head,
                         dim_head = config.dim_head,
                         ff_ratio = config.ff_ratio,
                         dropout_rate = config.dropout_rate,
                         device = config.device,
                         ex_ratio = config.ex_ratio)

    def forward(self, input_ids, **kwargs):
        logits = self.model(input_ids)
        return {"logits": logits}

# save files
config = MyModelConfig()
model = MyModelForCausalLM(config)
state_dict = torch.load("latest.pt", map_location="cpu")['model_state_dict']
state_dict = {f"model.{k}": v for k, v in state_dict.items()} # for meet HF wrap
model.load_state_dict(state_dict)

model.save_pretrained("/media/zhiyang/zhiyang_HDD/hf_upload/", safe_serialization=False) # safe_serialization=False is for weight tying.
tokenizer.save_pretrained("/media/zhiyang/zhiyang_HDD/hf_upload/")