import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.Preprocessor import Preprocessor
from utils.visualise import Visualisation
from utils.models import Model
from utils.map import Mapper

HARBOUR_DATA_PATH = "../../data/rawData/AllPortSedimentQuality.xlsx"
FAUNA_DATA_PATH = "../../data/rawData/Benthos families.xlsx"
SAVE_Data_PATH = "../../data/partPreprocessed/"

RETRIEVE_DATA_PATH = "../../data/preprocessed"

SAVE_RESULTS_PATH = "../../results/"


DO_PREPROCESSING = False
DO_VISULAIZATION = False
DO_MODELS = False
DO_MAP = True

if __name__ == "__main__":
    
    if DO_PREPROCESSING == True:
        preprocessing = Preprocessor(sedimentPath = HARBOUR_DATA_PATH, faunaPath = FAUNA_DATA_PATH, savePath=SAVE_Data_PATH)
        preprocessing.load_dataset()
    else:
        print("Skipped preprocessing")

    if DO_VISULAIZATION == True:
        visualisor = Visualisation(savePath= SAVE_RESULTS_PATH)
        # visualisor.correlationAnalysis()
        visualisor._compare_harbours()
    
    else:
        print("Skipped visualisation")
    
    if DO_MODELS == True:
        model = Model(path=RETRIEVE_DATA_PATH)
        model.linearModel()
        model.ER_Trees()
        model.randomForest()
        model.GBoostRegressor()
        model.NN()
    else:
        print("Skipped models")

    if DO_MAP == True:
        map = Mapper()
        map._make_map()
    else:
        print("Skipped mapping")