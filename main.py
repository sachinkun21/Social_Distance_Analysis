from utilities.human_detector import load_inference_graph
from utilities.human_detector import detect_objects

import os
from datetime import datetime
import cv2
from utilities import draw_bbox_cv2
import sys
from utilities import distance

# Default params
web_cam = False

path_vid = 'test/football.mp4'
im_height, im_width = (None, None)
score_thresh = 0.2
save_vid = True


def run_detections(web_cam, path_vid, im_height, im_width, score_thresh, save_vid):

    try:
        # Loading the Graph file
        print("----Loading the Model from path----")
        detection_graph, sess = load_inference_graph()

        if web_cam:
            vs = cv2.VideoCapture(0)


        else:
            vs = cv2.VideoCapture(path_vid)
            frame_width = int(vs.get(3))
            frame_height = int(vs.get(4))
            size = (frame_width, frame_height)


        if save_vid:
            result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

        num_frames = 0
        start_time = datetime.now()


        ## Let's start Reading Frame by Frame
        while True:
            ret, frame = vs.read()

            # for closing the loop frames end
            if not ret:
                break

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            print(" ---------------Running Detections--------------- ")

            # Run image through tensorflow graph
            boxes, scores, classes = detect_objects(frame, detection_graph, sess)

            num_frames += 1
            elapsed_time = (datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            #
            bbox_cords = draw_bbox_cv2.extract_box_and_plot(boxes, scores, classes,score_thresh, im_width, im_height, frame)
            distance.calc_dist_and_plot_close(frame, bbox_cords, im_height)

            #
            draw_bbox_cv2.draw_fps_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)

            cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if save_vid:
                result.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, "File Name: " , fname, "Line No: " , exc_tb.tb_lineno)


if __name__ == '__main__':
    run_detections(web_cam,path_vid,im_height,im_width, score_thresh,save_vid)
