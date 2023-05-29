import onnx
import onnxruntime as ort
import cv2
import tensorflow as tf
import numpy as np
import time

session = ort.InferenceSession('models/model_fp16.onnx', None)
input_name = session.get_inputs()[0].name
output_thresh = session.get_outputs()[0].name
output_coords = session.get_outputs()[1].name

cap = cv2.VideoCapture(0)
num_frames = 0
start_time = time.time()
total_inference_time = 0
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    img = np.expand_dims(resized/255,0)
    img = img.astype(np.float16)
    time1 = time.time()
    threshold = session.run([output_thresh], {input_name: img})
    coords = session.run([output_coords], {input_name: img})
    time2 = time.time()
    time3 = time2 - time1
    total_inference_time += time3  # Add current inference time to total inference time
    print(f"Inference time: {time3:.3f} s")
    output_threshh = np.array(threshold)
    output_coord = np.array(coords[0][0])
    yhat = output_threshh, output_coords

    
    if yhat[0] > 0.8: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(output_coord[:2], [450,450]).astype(int)),
                      tuple(np.multiply(output_coord[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(output_coord[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(output_coord[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(output_coord[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)

    num_frames += 1
    fps = num_frames / (time.time() - start_time)
    print(f'Frame rate: {fps:.2f} fps')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

avg_inference_time = total_inference_time / num_frames
print(f'Total number of frames: {num_frames}')
print(f'Total elapsed time: {time.time() - start_time:.2f} s')
print(f'Average FPS: {num_frames / (time.time() - start_time):.2f} fps')
print(f'Average inference time: {avg_inference_time:.2f} s')
