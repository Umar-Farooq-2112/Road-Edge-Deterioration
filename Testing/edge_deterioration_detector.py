import cv2
from edge_detector import road_edge_detector
from edge_deterioration import detect_deterioration
import numpy as np

def edge_deterioration_detector(image,mask,pixel_per_mm,edge_threshold=50,deterioration_threshold_mm = 80,line_method = 0,nam="a.png"):
    if image is None or mask is None:
        return None,None    
    left,right = road_edge_detector(image,mask,edge_threshold)
    w = int(mask.shape[1]/2)
    if left and right:
        left = detect_deterioration(mask[:,:w],pixel_per_mm,deterioration_threshold_mm,line_method,nam)
        right = detect_deterioration(cv2.flip(mask[:,w:],1),pixel_per_mm,deterioration_threshold_mm,line_method,nam)
    elif left:
        left = detect_deterioration(mask[:,:w],pixel_per_mm,deterioration_threshold_mm,line_method,nam)
        right = None
    elif right:
        left = None
        right = detect_deterioration(cv2.flip(mask[:,w:],1),pixel_per_mm,deterioration_threshold_mm,line_method,nam)
    else:
        return None,None
    return left, right


### Ignore the Code Below

# masks = []
# names = []
# scale = []
# for j in range(1,6):
#     for i in range (0,10):
#         names.append(f'images{j}/0000000{i}.png')
#         masks.append(f'masks{j}/0000000{i}.png')
#         scale.append(f'images{j}/imageScale.npy')
#     for i in range (0,6):
#         names.append(f'images{j}/0000001{i}.png')
#         masks.append(f'masks{j}/0000001{i}.png')
#         scale.append(f'images{j}/imageScale.npy')


# for j in range(3):
#     for i in range(len(names)):
#         image = cv2.imread(names[i])
#         mask = cv2.imread(masks[i])
#         mmperpx = np.load(scale[i])

#         print(f"{names[i]}:    {edge_deterioration_detector(image,mask,1/mmperpx,50,80,j,f'FinalResults/{j} {int(i/16)+1} {int(i%16)}.png')}")
