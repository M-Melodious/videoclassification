import cv2
import time
import argparse
import numpy as np

from pathlib import Path
from collections import deque

from filevideostream import FileVideoStream


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help="Path to input image file")
ap.add_argument('-v', '--video', help="Path to input video file")
ap.add_argument('-m', '--model', default="label_image/data/inception_v3_2016_08_28_frozen.pb",
                help="Path to frozen tensorflow graph")
ap.add_argument('-l', '--labels', default="label_image/data/imagenet_slim_labels.txt",
                help="Path to labels file")
ap.add_argument("-rs", "--resize", type=tuple, default=(299,299),
                help="Resize tuple, default (299,299)")
ap.add_argument("-im", "--mean", type=int, default=0, help="Input mean, default 0")
ap.add_argument("-is", "--std", type=int, default=255, help="Input std, default 255")
ap.add_argument("-qs", "--queue_size", type=int, default=25, help="Size of queue for averaging")


def load_tensorflow_graph(model_file:str):
    """
    Helper function to read the frozen tensorflow graph using
    opencv's dnn module.
    
    """
    model = cv2.dnn.readNetFromTensorflow(model_file)

    return model

def create_blob_from_image(img:np.ndarray, resize:tuple=(299,299),
                           input_mean:int=0, input_std:int=255):
    """
    Helper function to preprocess the input image, create blob, and return it.
    
    """
    mean = np.array([1.0, 1.0, 1.0]) * input_mean
    scale = 1 / input_std

    blob = cv2.dnn.blobFromImage(
        image=img,
        scalefactor=scale,
        size=resize,
        mean=mean,
        swapRB=True,
        crop=False
    )

    return blob

def load_labels(path_to_label_file:str):
    """
    Helper function to read the label file. Returns labels list.
    
    """
    labels = None
    with open(path_to_label_file, mode="r") as label_file:
        labels = label_file.readlines()

    return [label.rstrip() for label in labels]

def _classify_image(path_to_image, model, labels, args):
    """
    Main classifying function for image. Reads an image, makes an inference on
    it, and displays the prediction on image.
    
    """
    ## Read the input image and convert it to float32
    image = cv2.imread(path_to_image).astype(np.float32)

    ## Get the blob
    blob = create_blob_from_image(image,
                                  resize=args["resize"],
                                  input_mean=args["mean"],
                                  input_std=args["std"]
                                  )

    ## Set input and make prediction
    model.setInput(blob)
    preds = model.forward()

    ## Get the label and score
    class_id = np.argmax(preds)
    label = labels[class_id]
    score = preds[0][class_id] * 100

    ## Put text on image
    font_color = (0, 255, 0) if score > 80.0 else (0, 0, 255)
    text = f"Class: {label} Score: {score:.2f}%"
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, font_color, 3)

    ## Print and display the image
    print(f"{path_to_image} --> {text}")
    cv2.imshow("Image Classification", image)
    cv2.waitKey(0)
    

def _classify_video(file_video_stream, model,
                    labels, pred_queue, args):
    """
    Function to read the video file, classify each frame,
    rolling prediction average, display the video, and save the video.
    
    """
    while file_video_stream.more():
        ## Grab the frame from buffer
        frame = file_video_stream.read()

        if frame is None:
            break

        ## For display and write purpose
        output = cv2.resize(frame.copy(), (720, 600))

        ## Get the blob
        blob = create_blob_from_image(frame.astype(np.float32),
                                      resize=args["resize"],
                                      input_mean=args["mean"],
                                      input_std=args["std"])

        ## Set input, make inference, and then update the predictions
        ## queue
        model.setInput(blob)
        preds = model.forward()
        pred_queue.append(preds)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(pred_queue) == args["queue_size"]:
            ## Performing prediction average over the current history
            ## of previous predictions to reduce 'flickering'
            results = np.array(pred_queue).mean(axis=0)
            class_id = np.argmax(results)
            label = labels[class_id]
            pred_score = results[0][class_id] * 100

            font_color = (0, 255, 0) if pred_score > 80.0 else (0, 0, 255)
            text = f"Class: {label} Score: {pred_score:.2f}%"
            cv2.putText(output, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, font_color, 3)

            ## Print the label
            print(f"{args['video']} --> {text}")

        ## Display the output image
        cv2.imshow("Video Classification", output)
        key = cv2.waitKey(1) & 0xFF

        ## If the `q` was pressed, break from the loop
        if key == ord("q"):
            break

    ## Release the file pointers
    file_video_stream.stop()
    cv2.destroyAllWindows()

    return


def main(args):
    """
    Main entry point function.
    
    """
    ## First check for files existence
    assert Path(args["model"]).is_file(), "Frozen graph not found!"
    assert Path(args["labels"]).is_file(), "Label file not found!"

    ## Load frozen graph and labels
    print(f"Loading frozen graph ...")
    net = load_tensorflow_graph(args["model"])
    print(f"Loading labels ...")
    labels = load_labels(args["labels"])

    ## Either image or video
    if args["image"]:
        print(f"Predicting on image...")
        
        ## Check if image is present or not
        assert Path(args["image"]).is_file(), "Image not found!"
        _classify_image(args["image"], net, labels, args)
        
    elif args["video"]:
        print(f"Predicting on video...")
        
        ## Check if video is present or not
        assert Path(args["video"]).is_file(), "Video not found!"
        
        ## Prediction queue
        pred_queue = deque(maxlen=args["queue_size"])

        ## Initialize the file video stream
        fvs = FileVideoStream(args["video"], queue_size=args["queue_size"]).start()

        time.sleep(1.0) ## Allow the buffer to start to fill

        _classify_video(fvs, net, labels, pred_queue, args)
        
    else:
        print(f"Predicting default on image...")
        
        ## If neither, use default image, grace_hopper.jpg
        image = "label_image/test data/grace_hopper.jpg"
        _classify_image(image, net, labels, args)

    print("Done!")


if __name__ == '__main__':
    main(vars(ap.parse_args()))
