import cv2
import numpy as np

def count_votes(edges, rho, theta):
    edge_points = np.argwhere(edges != 0)
    
    edge_points = np.flip(edge_points, axis=1)
    
    distances = np.abs(edge_points[:, 0] * np.cos(theta) + edge_points[:, 1] * np.sin(theta) - rho)
    
    vote_count = np.sum(distances < 1.0)
    
    return vote_count





def hough_line(mask):
    edges = cv2.Canny(mask,100,300,apertureSize=3)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,int(mask.shape[0]/25))
    if lines is None:
        return False,(None,None)
    votes = np.array([count_votes(edges, rho, theta) for rho, theta in lines[:, 0]])
    
    max_votes = np.max(votes)
    max_votes_index = np.where(votes==max_votes)[0][0]
    
    res = lines[max_votes_index][0]
    rho, theta = res
   
    if np.sin(theta) == 0:
        return False,(rho,None)
            
    slope = - np.sin(theta)/np.cos(theta)
    intercept =  rho/np.cos(theta)

    return True,(slope,intercept)


