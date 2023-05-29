from tensorflow.keras.models import load_model
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
import tensorflow as tf 
import cv2
import numpy as np
import time

facetracker = load_model('models/facetracker.h5')



cap = cv2.VideoCapture(0)
num_frames = 0
start_time = time.time()
total_inference_time = 0

while cap.isOpened() and num_frames <=100:
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    img = np.expand_dims(resized/255,0)
    time1 = time.time()
    yhat = facetracker.predict(img)
    time2 = time.time()
    time3 = time2 - time1
    total_inference_time += time3  # Add current inference time to total inference time
    print(f"Inference time: {time3:.3f} s")
    print(yhat[1][0])
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.8: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
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

# Calculate and print the average inference time
avg_inference_time = total_inference_time / num_frames
print(f'Total number of frames: {num_frames}')
print(f'Total elapsed time: {time.time() - start_time:.2f} s')
print(f'Average FPS: {fps:.2f} fps')
print(f'Average inference time: {avg_inference_time:.2f} s')
