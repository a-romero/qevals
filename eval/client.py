import config
import json
from datasets import Dataset as ds
from datetime import datetime

from eval_tests import base_tests

eval_dataset = ds.from_csv('datagen/qac_out/eval_dataset_20240527-211903.csv')

gen_provider = config.config['DATAGEN']['GEN_PROVIDER']
eval_provider = config.config['EVAL']['EVAL_PROVIDER']
test_list = config.config['EVAL']['EVAL_TESTS']
test_results = base_tests(
    gen_provider=gen_provider,
    eval_provider=eval_provider,
    dataset=eval_dataset,
    test_list=test_list,
    use_answers_from_dataset=False
)
print(json.dumps(test_results, indent=2))

current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f"datagen/qac_out/eval_results_{current_date}.json", 'w') as f:
    json.dump(test_results, f)
