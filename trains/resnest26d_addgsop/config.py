import os
import sys
import inspect
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), '../../')))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), './')))
from models.ResNeSt_GSoP_Add import ResNeStGSoPUPnetPPModel

N_NAME = os.path.basename(os.path.dirname(inspect.getfile(inspect.currentframe())))

MODEL_INTERFACE = ResNeStGSoPUPnetPPModel
if __name__ == '__main__':
    model = ResNeStGSoPUPnetPPModel()
    print(model)