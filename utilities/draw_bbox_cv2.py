import cv2


def draw_rectangle(image_np, p1, p2, color_1,score,thickness):
    cv2.rectangle(image_np, p1, p2, color_1, thickness)

    # Label and Confidence value
    conf = round(score * 100, 2)
    cv2.putText(image_np, "Person {}%".format(conf), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)


def show_count_persons(image_np, count_p,im_height):
    cv2.putText(image_np, "Total Persons in Frame: {}".format(count_p), (10, im_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 100, 255), 2)


def show_close_persons(image_np, close_p, im_height):
    cv2.putText(image_np, "Close Persons in Frame: {}".format(close_p), (10, im_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 80, 0), 2)


def plot_close_lines(p1, p2, image_np):
    p1 = int(p1[0]),int(p1[1])
    p2 = int(p2[0]), int(p2[1])

    cv2.line(image_np, p1, p2, (255,255, 0), 2)


def extract_box_and_plot(boxes, scores, classes, score_thresh, im_width, im_height, image_np):
    # For Human
    color_1 = (0, 255, 0)
    bbox_coord = []

    for i in range(len(scores)):
        if scores[i]>score_thresh:
            if classes[i] == 1:
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                              boxes[i][0] * im_height, boxes[i][2] * im_height)
                bbox_coord.append((left, right, top, bottom, scores[i]))


    show_count_persons(image_np, len(bbox_coord), im_height)
    return bbox_coord


# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)




