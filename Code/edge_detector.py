import cv2
from basic_functions import binary_mask,apply_threshold,get_largest_contour

def road_edge_detector(image,mask,max_threshold):
        
    image = binary_mask(image)
    mask = binary_mask(mask)

    largest_contour_mask = get_largest_contour(mask)
    merged = cv2.bitwise_xor(image,largest_contour_mask)
    
    left_side = merged[:,:int(merged.shape[1]/2)]
    right_side = merged[:,int(merged.shape[1]/2):]

    left =  apply_threshold(left_side,max_threshold)
    right =  apply_threshold(right_side,max_threshold)

    return left,right
