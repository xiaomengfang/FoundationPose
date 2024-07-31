"""
File: PE.py
Author: xmfang
Description: interface for pose estimation 

History:
    - Version 0.0 (2024-07-31): xmfang, created

Dependencies:
    - in README.md
"""
import numpy as np
from estimater import *
from datareader import *

class PoseEstimation():
    
    def __init__(self, mesh_file:str):
        set_seed(0)

        code_dir = os.path.dirname(os.path.realpath(__file__))
        debug_dir = f'{code_dir}/debug'
        debug = 1
        self.mesh = trimesh.load(mesh_file)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        
        self.est = FoundationPose(model_pts=self.mesh.vertices, 
                                  model_normals=self.mesh.vertex_normals, 
                                  mesh=self.mesh, 
                                  scorer=scorer, 
                                  refiner=refiner, 
                                  debug_dir=debug_dir, 
                                  debug=debug, 
                                  glctx=glctx)
        
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.cnt = 0
        
    def inference(self, 
                  rgb:np.ndarray, 
                  depth:np.ndarray, 
                  mask:np.ndarray, 
                  K:np.ndarray, 
                  is_vis:bool=False,
                  ):
        if depth.mean() > 10.0:
            depth = depth * 0.001
            # depth = depth.astype(np.float64)
        if self.cnt == 0:
            pose = self.est.register(K=K, 
                                     rgb=rgb, 
                                     depth=depth, 
                                     ob_mask=mask, 
                                     iteration=self.est_refine_iter)
        else:
            pose = self.est.track_one(rgb=rgb, 
                                      depth=depth, 
                                      K=K, 
                                      iteration=self.track_refine_iter)
        self.cnt += 1
        
        if is_vis:
            to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, 
                                transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(0)
        
        return pose
    
if __name__ == "__main__":
    
    import time
    
    ts = time.time()
    PE = PoseEstimation(mesh_file='/home/ps/Projects/FoundationPose/demo_data/tote_transfering/box/mesh/box.obj')
    te = time.time()
    print(f'Initialization time: {te-ts:.3f}s')
    
    color = imageio.imread('/home/ps/Projects/FoundationPose/demo_data/tote_transfering/box/rgb/000000.png')
    depth = imageio.imread('/home/ps/Projects/FoundationPose/demo_data/tote_transfering/box/depth/000000.png')
    mask = imageio.imread('/home/ps/Projects/FoundationPose/demo_data/tote_transfering/box/masks/000000.png')
    K = np.array([607.1722412109375, 0, 319.3473205566406, 0, 607.257080078125, 253.62425231933594, 0, 0, 1]).reshape(3,3)
    ts = time.time()
    pose = PE.inference(rgb=color, depth=depth, mask=mask, K=K, is_vis=False)
    te = time.time()
    print(f'Inference time: {te-ts:.3f}s')
    print(f'pose:{pose}')

# pose:
# array([[ 0.01934849,  0.0337849 ,  0.99924177, -0.11960182],
#        [-0.8680746 , -0.49529833,  0.03355495,  0.06729078],
#        [ 0.4960564 , -0.86806566,  0.01974454,  0.96264017],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]],
#       dtype=float32)