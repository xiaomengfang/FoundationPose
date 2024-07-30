
import numpy as np
from estimater import *
from datareader import *

class FoundationPose():
    
    def __init__(self, model_pts, model_normals, mesh, scorer, refiner, debug_dir, debug, glctx):
        self.code_dir = os.path.dirname(os.path.realpath(__file__))
        
        set_seed(0)
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        
        
  


    def inference(self, rgb:np.ndarray, depth:np.ndarray, mask:np.naarray, K:np.ndarray, iteration:int):
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        