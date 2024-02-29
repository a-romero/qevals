import config
from datasets import Dataset as ds
from datetime import datetime

from datagen import create_synthetic_data

data_corpus_dir = config.config['DATAGEN']['DATA_DIR']
gen_provider = config.config['DATAGEN']['GEN_PROVIDER']

synthetic_dataset = create_synthetic_data(data_corpus_dir=data_corpus_dir,
                                          gen_provider=gen_provider)

print(synthetic_dataset[0])

current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
synthetic_dataset.to_csv(f"./datagen/qac_out/eval_dataset_{current_date}.csv")
