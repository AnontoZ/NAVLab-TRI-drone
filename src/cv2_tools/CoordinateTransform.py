import numpy as np 

# Converts from pixel frame to world frame
def pixelToWorld(px, cam_mtx, R, t, z_cf:float = 1, ):
    '''
    Description: Function to convert from pixel frame to world frame
    Inputs:
        - px: 2-by-n matrix of pixels coordinates to convert
        - cam_mtx: 3-by-3 camera matrix
        - R: 3-by-3 rotation matrix from world frame to camera frame
        - t: 3-by-1 translation from world frame to camera frame
    Outputs:
        - pts_wf: 3-by-n matrix of coordinates in world frame
    '''

    pts_cf = np.ones((3,px.shape[1]))
    # Convert from pixel to camera frame using camera matrix 
    pts_cf[0:2,:] = (px[0:2,:] - np.array([[cam_mtx[0,2]],[cam_mtx[1,2]]]))*(z_cf/np.array([[cam_mtx[0,0]],[cam_mtx[1,1]]]))
    pts_cf[2,:] = z_cf

    # Convert from camera frame to world frame 
    pts_wf = np.linalg.inv(R)@(pts_cf - t.reshape(3,1))
    
    return pts_wf 