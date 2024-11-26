import cv2
from ultralytics import YOLO
import torch
from get_data import GetData
import numpy as np

def RunYOLO():
    '''
    Instaniates YOLO model
    Runs the images over the model
    Outputs data
    Outputs images with bboxes

    Note: Commented portion is unfinished and for videos, but we should not need this.
    '''
    
    ####### For images: ##########

    # Initialize list and object for gathering bbox data
    ID_list = torch.tensor([])
    data_obj = GetData()

    # Load the YOLOv8 model
    model = YOLO('src/models/'+ 'yolov8s-worldv2.pt')

    # Run the model on a list JPG images. Live feed will replace this list and loop
    img_list = ['IMG_8567.JPG','IMG_8568.JPG','IMG_8569.JPG','julian.jpg']
    for idx, img in enumerate(img_list):
        results = model.track('src/YOLO/input_media/' + img)
        ID_list = data_obj.AddGetData(results = results, ID_list= ID_list)
        
        output_image = results[0].plot()
        cv2.imwrite(f'src/YOLO/output_media/' + f'output_image{idx}.jpg', output_image)

    # Save results
    ID_list[:, :2] = ID_list[:, :2].int()
    np.savetxt('src/YOLO/output_data/' + 'bbox_data.txt', ID_list, fmt='%f')

    ####### For live video feed: ##########

    # # Load the YOLO11 model
    # model = YOLO("yolo11n.pt")

    # # Open the video file
    # video_path = "src/YOLO/video.mp4"
    # cap = cv2.VideoCapture(video_path)

    # # Define the codec and create a VideoWriter object to save the video
    # # Use the same frame width and height as the input video
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    # output_path = "src/YOLO/annotated_output.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' codec for mp4 files
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # # Loop through the video frames
    # while cap.isOpened():
    #     # Read a frame from the video
    #     success, frame = cap.read()

    #     if success:
    #         # Run YOLO11 tracking on the frame, persisting tracks between frames
    #         results = model.track(frame, persist=True)

    #         # Visualize the results on the frame
    #         annotated_frame = results[0].plot()

    #         # Write the frame to the output video
    #         out.write(annotated_frame)

    #         # Display the annotated frame
    #         cv2.imshow("YOLO11 Tracking", annotated_frame)

    #         # Break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
    #     else:
    #         # Break the loop if the end of the video is reached
    #         break

    # # Release the video capture and writer objects and close the display window
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    RunYOLO()