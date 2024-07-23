import numpy as np
import cv2
from basic_functions import binary_mask,crop_required_image,first_nonzero_indices,get_largest_contour,best_fit_line
from hough_transform import hough_line
from runsat_line import runsat_line

def perpendicular_distance(slope, intercept, point):
    x1, y1 = point
    numerator = abs(slope * x1 - y1 + intercept)
    denominator = np.sqrt(slope**2 + 1)    
    distance = numerator / denominator
    
    return distance


def get_difference_values(mask,coefficients,status):
    indices = first_nonzero_indices(mask)
    difference = []
    if status:
        line_function = np.poly1d(coefficients)
        for i in range(len(mask)):
            if indices[i]!=-1:
                difference.append(perpendicular_distance(line_function.coefficients[0],line_function.coefficients[1],(i,indices[i])))
    else:
        for i in range(len(mask)):
            if indices[i]!=-1:
                difference.append(coefficients[0]-indices[i])
        
    return difference



# Line Methods
# 1--> Hough Transform
# 2--> RunSAT
# Else --> Numpy Polyfit



def detect_deterioration(image,pixel_per_mm,deterioration_threshold_mm=80,line_method = 0):
    mask = binary_mask(image)
    mask,_,_ = crop_required_image(mask)
    largest_contour_mask = get_largest_contour(mask)    
    
    
       
    if line_method == 1:
        status,coefficients = hough_line(largest_contour_mask.copy())
    elif line_method == 2:
        status,coefficients = runsat_line(largest_contour_mask.copy())
    else:
        status,coefficients = best_fit_line(largest_contour_mask.copy())
          
    if not status and coefficients[0] is None and coefficients[1] is None:
        return False 
        
    difference = get_difference_values(largest_contour_mask,coefficients,status)    
    
    height = mask.shape[0]
    
    if pixel_per_mm is None:
        pixel_per_mm = height/10000

    deterioration_threshold = deterioration_threshold_mm*pixel_per_mm    

    return np.max(difference)>(deterioration_threshold)