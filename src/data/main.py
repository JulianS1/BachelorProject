import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.Preprocessor import Preprocessor
from utils.visualise import Visualisation
from utils.models import Model

HARBOUR_DATA_PATH = "../../data/rawData/AllPortSedimentQuality.xlsx"
SAVE_Data_PATH = "../../data/partPreprocessed/"

SAVE_RESULTS_PATH = "../../results/"


DO_PREPROCESSING = True
DO_VISULAIZATION = False

if __name__ == "__main__":
    
    if DO_PREPROCESSING == True:
        preprocessing = Preprocessor(sedimentPath = HARBOUR_DATA_PATH, savePath=SAVE_Data_PATH)
        preprocessing.load_dataset()
    else:
        print("Skipped preprocessing")

    if DO_VISULAIZATION == True:
        visualisor = Visualisation(savePath= SAVE_RESULTS_PATH)
        visualisor.correlationAnalysis()
    
    else:
        print("Skipped visualisation")
    
    # model = Model(path=SAVE_Data_PATH)
    # model.linearModel()
    # model.randomForest()
    # model.GBoostRegressor()
    # model.NN()