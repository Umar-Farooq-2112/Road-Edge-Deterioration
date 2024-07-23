import numpy as np
import cv2
import random

def count_inliers(points, line_params, threshold, vertical_line=False, x_val=None):
    if vertical_line: 
        inliers = 0
        for x, y in points:
            if abs(x - x_val) < threshold:
                inliers += 1
        return inliers
    else:
        a, b = line_params
        inliers = 0
        for x, y in points:
            distance = abs(a * x - y + b) / np.sqrt(a**2 + 1)
            if distance < threshold:
                inliers += 1
        return inliers

def fit_line(points):
    (x1, y1), (x2, y2) = points
    if x1 == x2:
        return None, True, x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return (slope, intercept), False, None

def runsat_line(mask, num_iterations=100, threshold=2):
    edges = cv2.Canny(mask, 100, 300, apertureSize=3)
    edge_points = np.argwhere(edges > 0)
    
    best_line = None
    max_inliers = 0
    vertical_line = False
    x_val = None
    
    for _ in range(num_iterations):
        sample_points = random.sample(list(edge_points), 2)
        candidate_line, is_vertical, candidate_x_val = fit_line(sample_points)
        
        inliers = count_inliers(edge_points, candidate_line, threshold, vertical_line=is_vertical, x_val=candidate_x_val)
        
        if inliers > max_inliers:
            max_inliers = inliers
            best_line = candidate_line
            vertical_line = is_vertical
            x_val = candidate_x_val
    
    if best_line is None:
        return False, (None, None)
    
    if vertical_line:
        return False, (x_val, None)
    
    slope, intercept = best_line
    return True, (slope, intercept)