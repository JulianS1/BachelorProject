import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.Preprocessor import Preprocessor

SEDIMENT_DATA_PATH = "../../data/rawData/AllPortSedimentQuality.xlsx"
SAVE_PATH = "../../data/preprocessed/"

if __name__ == "__main__":
    
    preprocessing = Preprocessor(sedimentPath = SEDIMENT_DATA_PATH, savePath=SAVE_PATH)
    preprocessing.load_dataset()

    