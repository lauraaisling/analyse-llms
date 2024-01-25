# python scripts/test.py
import entropy_perplexity_utils as utils

json_path = "preset_configs/pythia6.9b-base.json"
config = utils.read_json(json_path)
model_type_test = config['model_type']
model_type = config.pop("model_type")
print("config ", config)
print("model_type ", model_type)
print("model_type_test ", model_type_test)