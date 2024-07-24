import cv2
import numpy as np
import random 

def best_fit_line(mask):        
    y = first_nonzero_indices(mask)
    x = (np.arange(len(y)))[y != -1]
    y = y[y!=-1]
    if np.all(y == y[0]):
        return False,(y[0],None)
    
    coefficients = np.polyfit(x, y, 1)
    return True,coefficients


def binary_mask(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = np.zeros_like(grayscale_image)
    binary_img[grayscale_image!= 0] = 255
    return binary_img
    

def apply_threshold(image,max_threshold):
    different_pixels = np.count_nonzero(image,axis = 1)
    return np.mean(different_pixels)>max_threshold


def crop_required_image(mask,crop_ratio=0.1):
    for i,item in enumerate(mask):
        if np.count_nonzero(item)!=0:
            break
    for j in range(mask.shape[0]-1,-1,-1):
        if np.count_nonzero(mask[j])!=0:
            break
    mask = mask[i:j,:]
    height = mask.shape[0]
    mask = mask[int(height*crop_ratio):height-int(height*crop_ratio),:]
    return mask,i,j


def first_nonzero_indices(arr):
    indices = np.argmax(arr != 0, axis=1)
    all_zero_rows = np.all(arr == 0, axis=1)
    indices[all_zero_rows] = -1
    return indices


def get_largest_contour(mask):
    res = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(res, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return res

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



def edgeDeteriorationDetector(image,mask,pixel_per_mm,edge_threshold=50,deterioration_threshold_mm = 80,line_method = 0):
    if image is None or mask is None:
        return None,None
    left,right = road_edge_detector(image,mask,edge_threshold)
    w = int(mask.shape[1]/2)
    if left and right:
        left = detect_deterioration(mask[:,:w],pixel_per_mm,deterioration_threshold_mm,line_method)
        right = detect_deterioration(cv2.flip(mask[:,w:],1),pixel_per_mm,deterioration_threshold_mm,line_method)
    elif left:
        left = detect_deterioration(mask[:,:w],pixel_per_mm,deterioration_threshold_mm,line_method)
        right = None
    elif right:
        left = None
        right = detect_deterioration(cv2.flip(mask[:,w:],1),pixel_per_mm,deterioration_threshold_mm,line_method)
    else:
        return None,None
    return left, right




### Ignore the Code Below
masks = []
names = []
scale = []
for j in range(1,6):
# j=5
    for i in range (0,10):
        names.append(f'Dataset/images{j}/0000000{i}.png')
        masks.append(f'Dataset/masks{j}/0000000{i}.png')
        scale.append(f'Dataset/images{j}/imageScale.npy')
    for i in range (0,6):
        names.append(f'Dataset/images{j}/0000001{i}.png')
        masks.append(f'Dataset/masks{j}/0000001{i}.png')
        scale.append(f'Dataset/images{j}/imageScale.npy')

for j in range(3):
    for i in range(len(names)):
        image = cv2.imread(names[i])
        mask = cv2.imread(masks[i])
        mmperpx = np.load(scale[i])
        print(f"{names[i]}:    {edgeDeteriorationDetector(image,mask,1/mmperpx,50,80,j)}")
