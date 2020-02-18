from src.data_ingestion import *
from src.processing import *
import os
import json
import sys



def main(args):
    if args[0] == 'data-test':
        cfg_path = os.getcwd() + '\config\\test-params.json'
        cfg = json.load(open(cfg_path))
    elif args[0] == 'data':
        cfg_path = os.getcwd() + '\config\data-params.json'
        cfg = json.load(open(cfg_path))
    database = create_database(**cfg)
    download_all_image(database)
    image_stats_database = create_dataframe()
    generate_stats(image_stats_database)


if __name__ == "__main__":
    main(sys.argv[1:])
