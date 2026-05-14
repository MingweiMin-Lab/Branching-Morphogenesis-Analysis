# Author: An Gong
# Version: 1.3
# Last modified: 2023-12-08

import numpy as np
from scipy.spatial.transform import Rotation as R
np.seterr(invalid='ignore') #silencing the RuntimeWarning "invalid value encountered in divide ray_direction = ray_direction / np.linalg.norm(ray_direction)"


def plot_bBox(viewer):
    scale = viewer.layers[-1].scale
    dataDim = viewer.layers[-1].data.shape
    X_DIM = dataDim[-1]*scale[-1]
    Y_DIM = dataDim[-2]*scale[-2]
    Z_DIM = dataDim[-3]*scale[-3]
    
    edges = np.array([[[0, 0, 0], [0, 0, X_DIM]],
     [[0, 0, 0], [0, Y_DIM, 0]],
     [[0, 0, 0], [Z_DIM, 0, 0]],
                  
     [[0, 0, X_DIM], [0, Y_DIM, X_DIM]],
     [[0, Y_DIM, X_DIM], [0,Y_DIM, 0]],          
     [[Z_DIM, 0, 0], [Z_DIM, 0, X_DIM]],
     [[Z_DIM, 0, X_DIM], [Z_DIM,Y_DIM, X_DIM]],
     [[Z_DIM,Y_DIM, X_DIM], [Z_DIM,Y_DIM, 0]],
     [[Z_DIM,Y_DIM, 0], [Z_DIM, 0, 0]],
     [[0, Y_DIM, 0], [Z_DIM, Y_DIM, 0]],
     [[0, Y_DIM, X_DIM], [Z_DIM, Y_DIM, X_DIM]],
     [[0, 0, X_DIM], [Z_DIM, 0, X_DIM]]], dtype = np.float64)

    if len(dataDim)==4:
        tDim = dataDim[0]
        edges_all=np.concatenate((np.zeros((12,2,1)), edges), axis=2)
        for t in range(1,tDim):
            a = np.concatenate((np.ones((12,2,1))*t, edges), axis=2)
            edges_all = np.concatenate((edges_all, a), axis = 0)
    else:
        edges_all=edges
    try:
        a = viewer.layers['b_box']
        a.data = edges_all
    except:
        box_layer=viewer.add_shapes(
            edges_all,
            shape_type='line',
            edge_color='white',
            edge_width = 1.2,
            face_color='white',
            scale = (1, 1, 1),
            # scale = viewer.layers[0].scale,
            opacity = 0.5,
            name = 'b_box',
        )
    
def plot_grid(viewer, interval):
    # interval has the unit after scaling

    scale = viewer.layers[-2].scale
    dataDim = viewer.layers[-2].data.shape
    X_DIM = dataDim[-1]*scale[-1]
    Y_DIM = dataDim[-2]*scale[-2]
    Z_DIM = dataDim[-3]*scale[-3]
    color = 'grey'
    line_width = 0.6
    opacity = 0.2

    # marks_x = np.arange(0, X_DIM, interval/scale[-1])
    # nXPoint = len(marks_x)
    # marks_y = np.arange(0, Y_DIM, interval/scale[-2]) 
    # nYPoint = len(marks_y)
    # marks_z = np.arange(0, Z_DIM, interval/scale[-3])
    # nZPoint = len(marks_z)

    marks_x = np.arange(0, X_DIM, interval)
    nXPoint = len(marks_x)
    marks_y = np.arange(0, Y_DIM, interval) 
    nYPoint = len(marks_y)
    marks_z = np.arange(0, Z_DIM, interval)
    nZPoint = len(marks_z)

    viewAngle = viewer.camera.angles
    r1 = R.from_euler('YZX', -np.array(viewAngle), degrees=True)
    rM = r1.as_matrix()
    ##the bottom grid
    #drawing the lines parallel to x axis
    bottom_Xgrid= np.zeros((nYPoint, 2, 3)) 
    bottom_Xgrid[:,0,0] = 0
    bottom_Xgrid[:,1,0] = 0
    bottom_Xgrid[:,0,1] = marks_y 
    bottom_Xgrid[:,1,1] = marks_y
    bottom_Xgrid[:,0,2] = 0
    bottom_Xgrid[:,1,2] = X_DIM
    #drawing the lines parallel to y axis
    bottom_Ygrid= np.zeros((nXPoint, 2, 3))
    bottom_Ygrid[:,0,0] = 0
    bottom_Ygrid[:,1,0] = 0
    bottom_Ygrid[:,0,1] = 0 
    bottom_Ygrid[:,1,1] = Y_DIM
    bottom_Ygrid[:,0,2] = marks_x
    bottom_Ygrid[:,1,2] = marks_x
    #combine X and Y lines
    bottom_grid = np.concatenate((bottom_Xgrid, bottom_Ygrid), axis=0)
    if rM[1, 2]>0:
        bottom_grid[:,:,0] = Z_DIM   
   
    ##the left grid
    #drawing the lines parallel to y axis
    left_Ygrid= np.zeros((nZPoint, 2, 3)) 
    left_Ygrid[:,0,0] = marks_z
    left_Ygrid[:,1,0] = marks_z
    left_Ygrid[:,0,1] = 0 
    left_Ygrid[:,1,1] = Y_DIM
    left_Ygrid[:,0,2] = 0
    left_Ygrid[:,1,2] = 0
    #drawing the lines parallel to z axis
    left_Zgrid= np.zeros((nYPoint, 2, 3))
    left_Zgrid[:,0,0] = 0
    left_Zgrid[:,1,0] = Z_DIM
    left_Zgrid[:,0,1] = marks_y
    left_Zgrid[:,1,1] = marks_y
    left_Zgrid[:,0,2] = 0
    left_Zgrid[:,1,2] = 0
    #combine X and Y lines
    left_grid = np.concatenate((left_Ygrid, left_Zgrid), axis=0)
    if rM[1,0]>0:
        left_grid[:,:,2] = X_DIM  
        
    ##the back grid
    #drawing the lines parallel to z axis
    back_Zgrid= np.zeros((nXPoint, 2, 3)) 
    back_Zgrid[:,0,0] = 0
    back_Zgrid[:,1,0] = Z_DIM
    back_Zgrid[:,0,1] = 0 
    back_Zgrid[:,1,1] = 0
    back_Zgrid[:,0,2] = marks_x
    back_Zgrid[:,1,2] = marks_x
    #drawing the lines parallel to x axis
    back_Xgrid= np.zeros((nZPoint, 2, 3))
    back_Xgrid[:,0,0] = marks_z
    back_Xgrid[:,1,0] = marks_z
    back_Xgrid[:,0,1] = 0
    back_Xgrid[:,1,1] = 0
    back_Xgrid[:,0,2] = 0
    back_Xgrid[:,1,2] = X_DIM
    #combine X and Z lines
    back_grid = np.concatenate((back_Xgrid, back_Zgrid), axis=0)
    if rM[1, 1]>0:
        back_grid[:,:,1] = Y_DIM  

    grids = np.concatenate((bottom_grid, left_grid, back_grid), axis=0)
    n_grids = grids.shape[0]
    if len(dataDim)==4:
        tDim = dataDim[0]
        grids_all=np.concatenate((np.zeros((n_grids,2,1)), grids), axis=2)
        for t in range(1,tDim):
            a = np.concatenate((np.ones((n_grids,2,1))*t, grids), axis=2)
            grids_all = np.concatenate((grids_all, a), axis = 0)
    else:
        grids_all = grids

    try:
        a = viewer.layers['grids']
        a.data = grids_all
        a.scale = scale
    except:
        layer = viewer.add_shapes(
            grids_all,
            shape_type = 'line',
            edge_color = color,
            opacity = opacity,
            edge_width = line_width,
            face_color = color,
            name='grids',
            scale = (1,1,1),
        )

def plot_ticks(viewer, interval):
    # interval has the unit after scaling
    scale = viewer.layers[-3].scale
    dataDim = viewer.layers[-3].data.shape
    X_DIM = dataDim[-1]*scale[-1]
    Y_DIM = dataDim[-2]*scale[-2]
    Z_DIM = dataDim[-3]*scale[-3]
    color = 'grey'
    line_width = 1
    opacity = 0.5    
    MINOR_TICK_LEN = np.round(np.minimum(X_DIM, Y_DIM)*0.05)
    MAJOR_TICK_LEN = MINOR_TICK_LEN*2
    
    # marks_x = np.arange(0, X_DIM, interval/(5*scale[-1]))
    # nXPoint = len(marks_x)
    # marks_y = np.arange(0, Y_DIM, interval/(5*scale[-2])) 
    # nYPoint = len(marks_y)
    # marks_z = np.arange(0, Z_DIM, interval/(5*scale[-3]))
    # nZPoint = len(marks_z)
    marks_x = np.arange(0, X_DIM, interval/5)
    nXPoint = len(marks_x)
    marks_y = np.arange(0, Y_DIM, interval/5) 
    nYPoint = len(marks_y)
    marks_z = np.arange(0, Z_DIM, interval/5)
    nZPoint = len(marks_z)
    viewAngle = viewer.camera.angles
    r1 = R.from_euler('YZX', -np.array(viewAngle), degrees=True)
    rM = r1.as_matrix()

    ##ticks at the edge parallel to x axis, tick1 has a bigger y value than tick2
    #the minor ticks
    xEdgeTick1_min= np.zeros((nXPoint, 2, 3)) 
    xEdgeTick2_min= np.zeros((nXPoint, 2, 3))
    #the major ticks
    xEdgeTick1_maj= np.zeros((np.ceil(nXPoint/5).astype(int), 2, 3)) 
    xEdgeTick2_maj= np.zeros((np.ceil(nXPoint/5).astype(int), 2, 3)) 

    xEdgeTick1_min[:,0,2] = marks_x 
    xEdgeTick1_min[:,1,2] = marks_x     
    xEdgeTick2_min[:,0,2] = marks_x 
    xEdgeTick2_min[:,1,2] = marks_x 
    xEdgeTick1_maj[:,0,2] = marks_x[::5] 
    xEdgeTick1_maj[:,1,2] = marks_x[::5]
    xEdgeTick2_maj[:,0,2] = marks_x[::5] 
    xEdgeTick2_maj[:,1,2] = marks_x[::5]

    xEdgeTick1_min[:,:,1] = Y_DIM
    xEdgeTick1_maj[:,:,1] = Y_DIM
    xEdgeTick2_min[:,:,1] = 0
    xEdgeTick2_maj[:,:,1] = 0
    
    if rM[1, 1]>0:     
        xEdgeTick2_min[:,0,1] = 0
        xEdgeTick2_min[:,1,1] = MINOR_TICK_LEN
        xEdgeTick2_maj[:,0,1] = 0
        xEdgeTick2_maj[:,1,1] = MAJOR_TICK_LEN
        if rM[1,2]>0:
            xEdgeTick2_min[:,:,0] = Z_DIM
            xEdgeTick2_maj[:,:,0] = Z_DIM
            
            xEdgeTick1_min[:,0,0] = 0
            xEdgeTick1_min[:,1,0] = MINOR_TICK_LEN
            xEdgeTick1_maj[:,0,0] = 0
            xEdgeTick1_maj[:,1,0] = MAJOR_TICK_LEN 
        else:
            xEdgeTick2_min[:,:,0] = 0
            xEdgeTick2_maj[:,:,0] = 0
            
            xEdgeTick1_min[:,0,0] = Z_DIM
            xEdgeTick1_min[:,1,0] = Z_DIM-MINOR_TICK_LEN
            xEdgeTick1_maj[:,0,0] = Z_DIM
            xEdgeTick1_maj[:,1,0] = Z_DIM-MAJOR_TICK_LEN    
    else:     
        xEdgeTick1_min[:,0,1] = Y_DIM
        xEdgeTick1_min[:,1,1] = Y_DIM - MINOR_TICK_LEN
        xEdgeTick1_maj[:,0,1] = Y_DIM
        xEdgeTick1_maj[:,1,1] = Y_DIM - MAJOR_TICK_LEN
        if rM[1,2]>0:
            xEdgeTick1_min[:,:,0] = Z_DIM
            xEdgeTick1_maj[:,:,0] = Z_DIM
            
            xEdgeTick2_min[:,0,0] = 0
            xEdgeTick2_min[:,1,0] = MINOR_TICK_LEN
            xEdgeTick2_maj[:,0,0] = 0
            xEdgeTick2_maj[:,1,0] = MAJOR_TICK_LEN 
        else:
            xEdgeTick1_min[:,:,0] = 0
            xEdgeTick1_maj[:,:,0] = 0
            
            xEdgeTick2_min[:,0,0] = Z_DIM
            xEdgeTick2_min[:,1,0] = Z_DIM-MINOR_TICK_LEN
            xEdgeTick2_maj[:,0,0] = Z_DIM
            xEdgeTick2_maj[:,1,0] = Z_DIM-MAJOR_TICK_LEN  

    xEdgeTick_edge = np.concatenate((xEdgeTick1_min, xEdgeTick1_maj, xEdgeTick2_min, xEdgeTick2_maj), axis=0)

    ##ticks at the edge parallel to y axis, tick1 has a bigger z value than tick2
    #the minor ticks
    yEdgeTick1_min= np.zeros((nYPoint, 2, 3)) 
    yEdgeTick2_min= np.zeros((nYPoint, 2, 3)) 
    #the major ticks
    yEdgeTick1_maj= np.zeros((np.ceil(nYPoint/5).astype(int), 2, 3)) 
    yEdgeTick2_maj= np.zeros((np.ceil(nYPoint/5).astype(int), 2, 3))

    yEdgeTick1_min[:,0,1] = marks_y 
    yEdgeTick1_min[:,1,1] = marks_y     
    yEdgeTick2_min[:,0,1] = marks_y 
    yEdgeTick2_min[:,1,1] = marks_y 
    yEdgeTick1_maj[:,0,1] = marks_y[::5] 
    yEdgeTick1_maj[:,1,1] = marks_y[::5]
    yEdgeTick2_maj[:,0,1] = marks_y[::5] 
    yEdgeTick2_maj[:,1,1] = marks_y[::5]    

   
    yEdgeTick1_min[:,:,0] = Z_DIM
    yEdgeTick1_maj[:,:,0] = Z_DIM
    yEdgeTick2_min[:,:,0] = 0
    yEdgeTick2_maj[:,:,0] = 0
    
    if rM[1, 2]>0:     
        yEdgeTick2_min[:,0,0] = 0
        yEdgeTick2_min[:,1,0] = MINOR_TICK_LEN
        yEdgeTick2_maj[:,0,0] = 0
        yEdgeTick2_maj[:,1,0] = MAJOR_TICK_LEN
        if rM[1,0]>0:
            yEdgeTick2_min[:,:,2] = X_DIM
            yEdgeTick2_maj[:,:,2] = X_DIM
            
            yEdgeTick1_min[:,0,2] = 0
            yEdgeTick1_min[:,1,2] = MINOR_TICK_LEN
            yEdgeTick1_maj[:,0,2] = 0
            yEdgeTick1_maj[:,1,2] = MAJOR_TICK_LEN 
        else:
            yEdgeTick2_min[:,:,2] = 0
            yEdgeTick2_maj[:,:,2] = 0
            
            yEdgeTick1_min[:,0,2] = X_DIM
            yEdgeTick1_min[:,1,2] = X_DIM-MINOR_TICK_LEN
            yEdgeTick1_maj[:,0,2] = X_DIM
            yEdgeTick1_maj[:,1,2] = X_DIM-MAJOR_TICK_LEN    
    else:     
        yEdgeTick1_min[:,0,0] = Z_DIM
        yEdgeTick1_min[:,1,0] = Z_DIM-MINOR_TICK_LEN
        yEdgeTick1_maj[:,0,0] = Z_DIM
        yEdgeTick1_maj[:,1,0] = Z_DIM-MAJOR_TICK_LEN
        if rM[1,0]>0:
            yEdgeTick1_min[:,:,2] = X_DIM
            yEdgeTick1_maj[:,:,2] = X_DIM
            
            yEdgeTick2_min[:,0,2] = 0
            yEdgeTick2_min[:,1,2] = MINOR_TICK_LEN
            yEdgeTick2_maj[:,0,2] = 0
            yEdgeTick2_maj[:,1,2] = MAJOR_TICK_LEN 
        else:
            yEdgeTick1_min[:,:,2] = 0
            yEdgeTick1_maj[:,:,2] = 0
            
            yEdgeTick2_min[:,0,2] = X_DIM
            yEdgeTick2_min[:,1,2] = X_DIM-MINOR_TICK_LEN
            yEdgeTick2_maj[:,0,2] = X_DIM
            yEdgeTick2_maj[:,1,2] = X_DIM-MAJOR_TICK_LEN           
    yEdgeTick_edge = np.concatenate((yEdgeTick1_min, yEdgeTick1_maj, yEdgeTick2_min, yEdgeTick2_maj), axis=0)
    

    ##ticks at the edge parallel to z axis, tick1 has a bigger x value than tick2
    #the minor ticks
    zEdgeTick1_min= np.zeros((nZPoint, 2, 3)) 
    zEdgeTick2_min= np.zeros((nZPoint, 2, 3)) 
    #the major ticks
    zEdgeTick1_maj= np.zeros((np.ceil(nZPoint/5).astype(int), 2, 3)) 
    zEdgeTick2_maj= np.zeros((np.ceil(nZPoint/5).astype(int), 2, 3))

    zEdgeTick1_min[:,0,0] = marks_z 
    zEdgeTick1_min[:,1,0] = marks_z     
    zEdgeTick2_min[:,0,0] = marks_z 
    zEdgeTick2_min[:,1,0] = marks_z 
    zEdgeTick1_maj[:,0,0] = marks_z[::5] 
    zEdgeTick1_maj[:,1,0] = marks_z[::5]
    zEdgeTick2_maj[:,0,0] = marks_z[::5] 
    zEdgeTick2_maj[:,1,0] = marks_z[::5]    

   
    zEdgeTick1_min[:,:,2] = X_DIM
    zEdgeTick1_maj[:,:,2] = X_DIM
    zEdgeTick2_min[:,:,2] = 0
    zEdgeTick2_maj[:,:,2] = 0

    if rM[1, 0]>0:     
        zEdgeTick2_min[:,0,2] = 0
        zEdgeTick2_min[:,1,2] = MINOR_TICK_LEN
        zEdgeTick2_maj[:,0,2] = 0
        zEdgeTick2_maj[:,1,2] = MAJOR_TICK_LEN
        if rM[1,1]>0:
            zEdgeTick2_min[:,:,1] = Y_DIM
            zEdgeTick2_maj[:,:,1] = Y_DIM
            
            zEdgeTick1_min[:,0,1] = 0
            zEdgeTick1_min[:,1,1] = MINOR_TICK_LEN
            zEdgeTick1_maj[:,0,1] = 0
            zEdgeTick1_maj[:,1,1] = MAJOR_TICK_LEN 
        else:
            zEdgeTick2_min[:,:,1] = 0
            zEdgeTick2_maj[:,:,1] = 0
            
            zEdgeTick1_min[:,0,1] = Y_DIM
            zEdgeTick1_min[:,1,1] = Y_DIM-MINOR_TICK_LEN
            zEdgeTick1_maj[:,0,1] = Y_DIM
            zEdgeTick1_maj[:,1,1] = Y_DIM-MAJOR_TICK_LEN    
    else:     
        zEdgeTick1_min[:,0,1] = Y_DIM
        zEdgeTick1_min[:,1,1] = Y_DIM-MINOR_TICK_LEN
        zEdgeTick1_maj[:,0,1] = Y_DIM
        zEdgeTick1_maj[:,1,1] = Y_DIM-MAJOR_TICK_LEN
        if rM[1,1]>0:
            zEdgeTick1_min[:,:,1] = Y_DIM
            zEdgeTick1_maj[:,:,1] = Y_DIM
            
            zEdgeTick2_min[:,0,1] = 0
            zEdgeTick2_min[:,1,1] = MINOR_TICK_LEN
            zEdgeTick2_maj[:,0,1] = 0
            zEdgeTick2_maj[:,1,1] = MAJOR_TICK_LEN 
        else:
            zEdgeTick1_min[:,:,1] = 0
            zEdgeTick1_maj[:,:,1] = 0
            
            zEdgeTick2_min[:,0,1] = Y_DIM
            zEdgeTick2_min[:,1,1] = Y_DIM-MINOR_TICK_LEN
            zEdgeTick2_maj[:,0,1] = Y_DIM
            zEdgeTick2_maj[:,1,1] = Y_DIM-MAJOR_TICK_LEN           
    zEdgeTick_edge = np.concatenate((zEdgeTick1_min, zEdgeTick1_maj, zEdgeTick2_min, zEdgeTick2_maj), axis=0)


    ticks = np.concatenate((xEdgeTick_edge, yEdgeTick_edge, zEdgeTick_edge), axis = 0)
    n_ticks = ticks.shape[0]
    if len(dataDim)==4:
        tDim = dataDim[0]
        ticks_all=np.concatenate((np.zeros((n_ticks,2,1)), ticks), axis=2)
        for t in range(1,tDim):
            a = np.concatenate((np.ones((n_ticks,2,1))*t, ticks), axis=2)
            ticks_all = np.concatenate((ticks_all, a), axis = 0)
    else:
        ticks_all = ticks    
    
    try:
        a = viewer.layers['ticks']
        a.data = ticks
        a.scale = scale
    except:
        layer2 = viewer.add_shapes(
            ticks,
            shape_type = 'line',
            edge_color = color,
            opacity = opacity,
            edge_width = line_width,
            face_color = color,
            name='ticks',
            scale = (1,1,1),
        )
    
# def plot_bSurf(viewer):
#         # interval has the unit after scaling
#     dataDim = viewer.layers[0].data.shape
#     X_DIM = dataDim[-1]
#     Y_DIM = dataDim[-2]
#     Z_DIM = dataDim[-3]    
#     bottom = np.array([[[Z_DIM,0,0], [Z_DIM,Y_DIM,0],[Z_DIM,Y_DIM,X_DIM], [Z_DIM,0, X_DIM]]], dtype = np.float64)
#     back = np.array([[[0,0,0], [0,0,X_DIM],[Z_DIM,0,X_DIM], [Z_DIM,0, 0]]], dtype = np.float64)
#     left = np.array([[[0,0,0], [Z_DIM,0,0],[Z_DIM,Y_DIM,0], [0,Y_DIM, 0]]], dtype = np.float64)
#     bounding=np.concatenate((bottom, back, left), axis=0)

#     try:
#         a = viewer.layers['b_surface']
#         a.data = bounding
#         a.scale = viewer.layers[0].scale
#     except:    
#         surf_layer = viewer.add_shapes(
#             bounding,
#             shape_type='rectangle',
#             face_color='grey',
#             # edge_color='black',
#             edge_width=0,
#             name = 'b_surface',
#             scale = (1,1,1),
#             opacity = 0.1,
#         )