import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import Annotator


pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
pipeline.start(config)



def cube_detection(pipeline):
    model = YOLO("/home/kavin/galileo2023/autonomous_codes/models/weights/best.pt")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        results = model.predict(img, conf = 0.75, max_det = 3)
        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

                # Get dep/home/kavin/caesar2020_nvidia/src/navigation/scripts/kavin_modifi_2.pyth data for the bounding box
                left, top, right, bottom = map(int, b)
                arrow_center = (left+right)/2
                try:
                    depth = depth_frame.get_distance((left + right) // 2, (top + bottom) // 2)
                except:
                    ret = False
                print("Depth for box" + str(b) + ":" + str(depth) +"meters")

                # Check if the arrow is in the center along the x-axis
              #  img_center_x = 320
                # img_center_x = img.shape[1] // 2
                # Calculate the center along x-axis
                #if left <= img_center_x <= right:
                #print("arrow coordinates relative to center ", arrow_center-img_center_x)
              #  if ((arrow_center-img_center_x)>-150) and (arrow_center - img_center_x)<125 or depth<3 :
                # if depth<25:
              #      arrow_in_center = True
               # else:
                 #   arrow_in_center = False
                #print(arrow_center-img_center_x)
        # print("imshow")
        cv2.imshow('YOLO V8 Detection', img)   
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break         
        
        #print("arrow in center: ", arrow_in_center)  
        #if arrow_in_center:
            #cv2.putText(img, "Arrow in centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Depth from more than 3:", depth)
            return arrow_in_center, "Not Available", arrow_center, depth
        #else:
            return False, "Not available", None, 2.5
        
cube_detection(pipeline)