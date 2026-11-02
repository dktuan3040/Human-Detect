
import cv2
import numpy as np
import time

from processing.gaussian import gaussian_blur

from processing.morphology import opening

from detection.motion import detect_motion
from detection.motion import calculate_motion_ratio
from detection.contours import find_connected_components
from detection.contours import merge_near_boxes
from detection.person import is_person, split_by_gap


def to_grayscale(frame):
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]

    grayscale = (
        0.114 * blue +
        0.587 * green +
        0.299 * red
    )

    return grayscale.astype(np.uint8)


# ==============================
# CẤU HÌNH
# ==============================

camera = cv2.VideoCapture(0)

MOTION_THRESHOLD = 30    #Độ lệch pixel để bt nó có chuyển động >x
is_static = False          #Có đang tĩnh không
MOTION_RATIO = 2       #Tỉ lệ điều kiện là ảnh đọng hay tĩnh




start_time = None
start_bg_time = None
STATIC_WAIT_TIME = 5
background_ready = False
    
background_blurred = None    
MIN_AREA = 2000
LIGHT_RATIO = 60   # nếu hơn x% pixel khác nền -> coi là ánh sáng đổi
LIGHT_RATIO_MIN = 5   # nếu lớn x% pixel khác nền -> có chuyển động

LIGHT_WAIT_TIME = 30      # ánh sáng đổi liên tục xs mới reset nền
light_start_time = None   # mốc bắt đầu đếm ánh sáng đổi

# ==============================
# VÒNG LẶP CAMERA
# ==============================
previous_frame = None
while True:
    current_time = time.time()
    ret, frame = camera.read()

    if not ret:
        print("Khong doc duoc Camera")
        break
#1 Bộ lọc 
    grayscale_frame = to_grayscale(frame)
    blurred = gaussian_blur(grayscale_frame)

  
#Phát hiện motion
    if previous_frame is None:
        previous_frame = blurred
        continue
    motion_mask = detect_motion(previous_frame,blurred,MOTION_THRESHOLD)
    previous_frame = blurred

    
#Lọc lại 
    opened = opening(motion_mask)
#Ktra là Tĩnh hay động
    
    motion_ratio = calculate_motion_ratio(opened)
    print("Motion ratio:", motion_ratio)
    if motion_ratio > MOTION_RATIO:
        is_static = False
        start_time = None
        
    else:
        if start_time is None:
            start_time = current_time

        elif current_time - start_time >= STATIC_WAIT_TIME:
            is_static = True


#Xử lý BackGround 
    if not background_ready:
     if is_static:
        if start_bg_time is None:
            start_bg_time = time.time()
        elif time.time() - start_bg_time >= STATIC_WAIT_TIME:
            background_float = blurred.astype(np.float32)
            background_blurred = blurred.copy()
            background_ready = True
            start_bg_time = None
            print("Đã tạo background")
    
#Bắt đâu khi có Background
    if background_ready:
        foreground_mask = detect_motion(background_blurred, blurred, MOTION_THRESHOLD)
        foreground_opened = opening(foreground_mask)
        bg_ratio = calculate_motion_ratio(foreground_opened)
        print("Bacground ratio : ",bg_ratio)
            
        # Nếu quá nhiều pixel khác -> ánh sáng đổi -> bù sáng cho nền rồi so lại
        if bg_ratio > LIGHT_RATIO:
            print(f"hơn 60% do as môi trường")
            diff = blurred.astype(np.int16) - background_blurred.astype(np.int16)
            shift = int(np.median(diff))
            adjusted_bg = np.clip(background_blurred.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            if light_start_time is None:
                light_start_time = current_time
            elif current_time - light_start_time >= LIGHT_WAIT_TIME and is_static == True:
                background_blurred = blurred
                light_start_time = None
        
        # Lần 2: so với nền đã bù sáng
            foreground_mask = detect_motion(adjusted_bg, blurred, MOTION_THRESHOLD)
            foreground_opened = opening(foreground_mask)

        else: light_start_time = None     
        if is_static == False : light_start_time = None
        components = find_connected_components(foreground_opened)

        boxes = []
        for x, y, w, h in components:
            if w * h >= MIN_AREA:
                boxes.append((x, y, w, h))

        boxes = merge_near_boxes(boxes)   # gộp các ô gần nhau
        boxes = [p for b in boxes for p in split_by_gap(foreground_opened, b)]
        people = [b for b in boxes if is_person(foreground_opened, b, frame.shape)]
        for x, y, w, h in people:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        cv2.imshow("Foreground", foreground_opened)
         
#Show
    cv2.imshow("Camera", frame)
    
    print("Is static:", is_static)
    
    if background_ready:  
        cv2.imshow("Background", background_blurred)
  
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break




camera.release()
cv2.destroyAllWindows()
