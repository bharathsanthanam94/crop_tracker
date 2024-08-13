import cv2
import numpy as np
import sys
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import sys
import ipdb
import argparse
import yaml
import os
from tracker.byte_tracker import BYTETracker 
from tracker.utils import plot_tracking

def preprocess_image(src, input_w, input_h):
    scale = min(input_w / src.shape[1], input_h / src.shape[0])
    unpad_w = int(scale * src.shape[1])
    unpad_h = int(scale * src.shape[0])
    
    resized = cv2.resize(src, (unpad_w, unpad_h))
    preprocessed_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    preprocessed_img[:unpad_h, :unpad_w, :] = resized


    # Flatten the image into input tensor in RGB format
    input_tensor = np.transpose(preprocessed_img, (2, 0, 1)).astype(np.float32).flatten()
    return input_tensor,scale, unpad_w,unpad_h

def do_inference(engine, input_tensor, output_tensor):
    context = engine.create_execution_context()
    # find memory size for input and output
    input_size = trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size * np.dtype(np.float32).itemsize
    output_size = trt.volume(engine.get_binding_shape(1)) * engine.max_batch_size * np.dtype(np.float32).itemsize
    
    # Allocate device memory
    d_input = cuda.mem_alloc(input_tensor.nbytes)
    d_output = cuda.mem_alloc(output_tensor.nbytes)
    bindings = [int(d_input), int(d_output)]
    
    stream = cuda.Stream()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, input_tensor, stream)
    # Execute the model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(output_tensor, d_output, stream)
    # Synchronize stream
    stream.synchronize()
    
    # Clean up memory
    d_input.free()
    d_output.free()
    stream = None
    return output_tensor 

# Logger for TensorRT
class Logger(trt.Logger):
    def log(self, severity, msg):
        if severity <= trt.Logger.ERROR:
            print(msg)

def load_engine_from_file(file_name):
    TRT_LOGGER = Logger()
    with open(file_name, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def decode_predictions(outputs, strides,SINGLE_PRED_SIZE,CONF_THRESHOLD,input_h,input_w ):
    """
    Decode output from the network
    """
    grid_and_stride = []
    for stride in strides:
        grid_h = input_h // stride
        grid_w = input_w // stride
        for g1 in range(grid_h):
            for g0 in range(grid_w):
                grid_and_stride.append((g0, g1, stride))
    
    
    proposals = []
    for idx, (g0, g1, stride) in enumerate(grid_and_stride):
        
        picked_boxes= {}
        basic_pos= idx * (SINGLE_PRED_SIZE)
        x_center = (outputs[basic_pos+0] + g0) * stride
        y_center = (outputs[basic_pos+1]+g1) * stride
        w = np.exp(outputs[basic_pos+2]) * stride
        h = np.exp(outputs[basic_pos+3]) * stride

        # Top left at bounding box
        x0 = x_center-w*0.5
        y0= y_center -h*0.5

        box_objectness = outputs[basic_pos+4]

        for class_idx in range(2):
            box_cls_score = outputs[basic_pos+5+class_idx]
            box_prob = box_objectness*box_cls_score
            if box_prob > CONF_THRESHOLD:
                picked_boxes["x"]=x0
                picked_boxes["y"]=y0
                picked_boxes["width"]=w
                picked_boxes["height"]=h
                picked_boxes["class_id"]=class_idx
                picked_boxes["confidence"]=box_prob
                proposals.append(picked_boxes)
    
    return proposals
def iou(box1, box2):
    # Calculate the coordinates of the intersection 
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
    
    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of both bounding boxes
    box1_area = box1["width"] * box1["height"]
    box2_area = box2["width"] * box2["height"]
    
    # Calculate union area by using the formula: Union(A,B) = A + B - Intersect(A,B)
    union_area = box1_area + box2_area - intersection_area
    
    #  IoU
    return intersection_area / union_area if union_area != 0 else 0
      

def non_max_suppression(proposals, iou_threshold=0.45):
    # Sort proposals by confidence score in descending order
    sorted_proposals = sorted(proposals, key=lambda x: x["confidence"], reverse=True)
    keep = []  # This will store the indexes of the proposals to keep
    
    # Loop through each proposal
    while sorted_proposals:
        # Take the proposal with the highest confidence score
        current = sorted_proposals.pop(0)
        keep.append(current)
        
        # Compare this box with all remaining boxes
        sorted_proposals = [
            box for box in sorted_proposals if iou(current, box) < iou_threshold
        ]
    
    return keep


def draw_bounding_boxes_opencv(src_img, predictions, output_path):
    # Load the image
    img = src_img

    # Iterate through all bounding boxes
    for bbox in predictions:
        # Define colors in BGR format
        color = (255, 0, 0) if bbox['class_id'] == 0 else (0, 0, 255)  # Blue for Crop, Red for Weed

        # Calculate the top-left and bottom-right coordinates
        top_left = (int(bbox['x']), int(bbox['y']))
        bottom_right = (int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height']))

        # Draw the rectangle with the specified color
        cv2.rectangle(img, top_left, bottom_right, color, 2)

       
        class_name = "Crop" if bbox['class_id'] == 0 else "Weed"
        label = f"{class_name}: {bbox['confidence']:.2f}"
        cv2.putText(img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, img)
def scale_predictions(filtered_predictions,scale):
    scaled_predictions=[]
    for bbox in filtered_predictions:
        bbox_scaled={}
        x0 = bbox["x"]/scale
        y0 = bbox["y"]/scale
        x1= (bbox["x"]+bbox["width"])/scale
        y1= (bbox["y"]+bbox["height"])/scale

        img_width =1280
        img_height = 720
        #clip x0,y0,x1,y1 that are beyond the range of the image
        # Clip coordinates to ensure they stay within the image bounds
        x0 = max(0, min(x0, img_width))
        y0 = max(0, min(y0, img_height))
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))

        

        #
        bbox_scaled["x"]= x0
        bbox_scaled["y"]=y0
        bbox_scaled["width"]=x1-x0
        bbox_scaled["height"]=y1-y0
        bbox_scaled["class_id"]=bbox["class_id"]
        bbox_scaled["confidence"]=bbox["confidence"]
        scaled_predictions.append(bbox_scaled)
    return scaled_predictions

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments(config):
    parser = argparse.ArgumentParser("yolox_bytetrack")
    parser.add_argument("--track_thresh", type=float, default=config.get("track_thresh", 0.5), help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=config.get("track_buffer", 30), help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=config.get("match_thresh", 0.8), help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=config.get("mot20", False), action="store_true", help="test mot20.")
    parser.add_argument('--min_box_area', type=float, default=config.get("min_box_area", 10), help='filter out tiny boxes')
    parser.add_argument("--aspect_ratio_thresh", type=float, default=config.get("aspect_ratio_thresh", 1.6),
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument("--use_bbox_filters", default=config.get("use_bbox_filters", False), action="store_true",
                        help="use ByteTrack bbox size and dimensions ratio filters")
    parser.add_argument("--input_data_path", type=str, default=config.get("input_data_path", "/home/agrobot/agrobot-jetsons/cpp-TensorRT-inference/dataset_subset"))
    parser.add_argument("--save_det_path", type=str, default=config.get("save_det_path", "/home/agrobot/agrobot-jetsons/python-TensorRT-inference/inference_results/"))
    parser.add_argument("--engine_path", type=str, default=config.get("engine_path", "/home/agrobot/agrobot-jetsons/cpp-TensorRT-inference/models/yolox_m.engine"))
    parser.add_argument("--save_tracker_path", type=str, default=config.get("save_tracker_path", "/home/agrobot/agrobot-jetsons/python-TensorRT-inference/output_tracker.mp4"))
    parser.add_argument("--input_h",type=int,default=config.get("input_h",640))
    parser.add_argument("--input_w",type=int, default=config.get("input_w",640))
    parser.add_argument("--conf_threshold",type=float, default=config.get("conf_threshold",0.3))
    parser.add_argument("--nms_threshold",type=float, default=config.get("nms_threshold",0.45))
    parser.add_argument("--single_pred_size",type=int, default=config.get("single_pred_size",7))


    args = parser.parse_args()
    return args

def main():
    
    config = load_config('/home/agrobot/agrobot-jetsons/python-TensorRT-inference/config.yaml')
    args = parse_arguments(config)
    
    tracker = BYTETracker(args, frame_rate=4)
    # import ipdb;ipdb.set_trace()

    engine = load_engine_from_file(args.engine_path)
    img_path= args.input_data_path
    images= sorted(os.listdir(img_path))

    frame_id=0
    video_writer= cv2.VideoWriter(args.save_tracker_path,
                                    cv2.VideoWriter_fourcc('a', 'v', 'c', '1'),
                                    4,
                                    (1280,720))
    for img in images:


        src_img = cv2.imread(os.path.join(img_path,img))
        start_det = time.time()
        if src_img is None:
            print("Image load failed")
            return 1
        output_path=os.path.join(args.save_det_path, f"output_{frame_id}.jpg")
        # import ipdb;ipdb.set_trace()
        input_w, input_h = args.input_w, args.input_h 
        
        input_tensor,scale, unpad_w,unpad_h = preprocess_image(src_img, input_w, input_h)
        output_tensor = output_tensor = np.zeros((8400*7), dtype=np.float32)
        output = do_inference(engine,input_tensor,output_tensor)
        # ipdb.set_trace()
        # output_tensor = output.reshape(8400,7)
        scaled_img = cv2.resize(src_img, (input_w, input_h))
        strides= [8,16,32]
        predictions = decode_predictions(output,strides,args.single_pred_size, args.conf_threshold,args.input_h,args.input_w)
        filtered_predictions = filtered_proposals = non_max_suppression(predictions, iou_threshold=args.nms_threshold)
        stop_det = time.time()
        time_detector= stop_det - start_det
        print("total detection time:",time_detector)

        # Draw BBOX and save!
        scaled_predictions = scale_predictions(filtered_predictions,scale)
        draw_bounding_boxes_opencv(src_img, scaled_predictions,output_path)
        
        # tracker starts here!
        output_array = np.empty((len(filtered_predictions), 5))
        # Fill the array
        start_tracker = time.time()
        for i, bbox in enumerate(filtered_predictions):
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = x1 + bbox['width']
            y2 = y1 + bbox['height']
            confidence = bbox['confidence']
            output_array[i] = [x1, y1, x2, y2, confidence]
        
        img_info = [720,1280]
        img_size = [args.input_h,args.input_w]
        online_targets = tracker.update(output_array,img_info,img_size)
        end_tracker = time.time()
        time_tracker = end_tracker - start_tracker
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
        frame_fps=1/(time_detector+time_tracker)
        
        online_im = plot_tracking(
            src_img,
            online_tlwhs,
            online_ids,
            frame_id=frame_id + 1,
            fps=frame_fps
            )
    
        if online_im.shape[1] > 1280:
            show_image('frame', online_im, (1280, -1))
        else:
            cv2.imshow('frame', online_im)
        
        
        video_writer.write(online_im)
        frame_id+=1

    video_writer.release()
    cv2.destroyAllWindows()

       

if __name__ == "__main__":
    main()


	
