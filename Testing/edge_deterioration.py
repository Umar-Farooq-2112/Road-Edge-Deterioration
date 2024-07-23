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

def join_images_horizontally(images):
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    for i in range(len(images)):
        if images[i].shape[0] != max_height:
            # Resize image to have the same height as the tallest image
            scale = max_height / images[i].shape[0]
            new_width = int(images[i].shape[1] * scale)
            images[i] = cv2.resize(images[i], (new_width, max_height))

    joined_image = np.hstack(images)
    return joined_image


def detect_deterioration(image,pixel_per_mm,deterioration_threshold_mm=80,line_method = 0,nam="am.png"):
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
    
    edges = cv2.Canny(largest_contour_mask,200,300,apertureSize=3)
    
    
    if status:
        line_function = np.poly1d(coefficients)
        pt1 = (line_function(0).astype(int),0)
        pt2 = (line_function(height-1).astype(int),height-1)
        
    else:
        pt1 = (int(coefficients[0]),0)
        pt2 = (int(coefficients[0]),height-1)
    
    cv2.line(edges,pt1,pt2,255,1)
    w = mask.shape[1]
    joined_image = join_images_horizontally([mask,largest_contour_mask,edges])
    cv2.putText(joined_image, f"{np.max(difference)>(deterioration_threshold)}", (10,250), cv2.FONT_HERSHEY_SIMPLEX, 10, 150, 2, cv2.LINE_AA)
    cv2.putText(joined_image, f"Difference:{int(np.max(difference))}px", (w+10,200), cv2.FONT_HERSHEY_SIMPLEX, 5, 150, 2, cv2.LINE_AA)
    cv2.putText(joined_image, f"Threshold: {int(deterioration_threshold)}px", (10+(w),400), cv2.FONT_HERSHEY_SIMPLEX, 5, 150, 2, cv2.LINE_AA)
    cv2.putText(joined_image, f"Difference:{int(np.max(difference)/pixel_per_mm)}mm", (2*w+10,200), cv2.FONT_HERSHEY_SIMPLEX, 5, 150, 2, cv2.LINE_AA)
    cv2.putText(joined_image, f"Threshold: {int(deterioration_threshold/pixel_per_mm)}mm", (2*w+10,400), cv2.FONT_HERSHEY_SIMPLEX, 5, 150, 2, cv2.LINE_AA)
    cv2.imwrite(nam,joined_image)

    return np.max(difference)>(deterioration_threshold)
    
