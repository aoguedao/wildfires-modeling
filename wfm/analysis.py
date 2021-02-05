from pathlib import Path
from pprint import pprint

import wfm.preprocessing as pr

def main(path):
    input_path = path / "data" / "input"
    pr.validate_columns(input_path)
    input_data = pr.get_input_data(input_path)
    pprint(input_data.head())
    return input_data

if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent
    main(path)