
from utilities.draw_bbox_cv2 import show_close_persons
from utilities.draw_bbox_cv2 import plot_close_lines
from utilities.draw_bbox_cv2 import draw_rectangle



def euclidean_distance(p1, p2):
    return ((p2[0]-p1[0])**2+ (p2[1]-p1[1])**2)**0.5


def calculate_distance(image_np, bbox_cords):
    close_persons = []
    for i in range(len(bbox_cords)):
        person1 = bbox_cords[i]
        left_1, right_1, top_1, bottom_1, scores_1 = person1
        p1_centroid = (left_1 + right_1) // 2, (top_1 + bottom_1) // 2

        start_p1 = (int(left_1), int(top_1))
        end_p1 = (int(right_1), int(bottom_1))
        draw_rectangle(image_np, start_p1, end_p1, (0, 230, 0), scores_1, 3)

        for j in range( len(bbox_cords)):
            if i != j:
                person2 = bbox_cords[j]
                left_2, right_2, top_2, bottom_2, scores_2 = person2
                p2_centroid = (left_2 + right_2) // 2, (top_2 + bottom_2) // 2

                # Calculating pixel wise Distance between two persons
                dist = (euclidean_distance(p1_centroid, p2_centroid))

                if dist <= 160:
                    close_persons.append((p1_centroid, p2_centroid))
                    start_p1 = (int(left_1), int(top_1))
                    end_p1 = (int(right_1), int(bottom_1))
                    draw_rectangle(image_np, start_p1, end_p1, (255, 0, 0), scores_1, 3)

                    # #
                    # start_p2 = (int(left_2), int(top_2))
                    # end_p2 = (int(right_2), int(bottom_2))
                    # draw_rectangle(image_np, start_p2, start_p2, (255, 0, 0), scores_2, 3)

                # else:
                #     start_p1 = (int(left_1), int(top_1))
                #     end_p1 = (int(right_1), int(bottom_1))
                #     draw_rectangle(image_np, start_p1, end_p1, (0, 255, 0), scores_1, 3)
                    # start_p2 = (int(left_2), int(top_2))
                    # end_p2 = (int(right_2), int(bottom_2))
                    # draw_rectangle(image_np, start_p2, end_p2, (0, 255, 0), scores_1, 3)

    return close_persons

def calc_dist_and_plot_close(image_np, bbox_cords, im_height):
    # Stores centroids of Close Persons
    close_persons = calculate_distance(image_np, bbox_cords)
    # print(close_persons)
    # print(set(close_persons))

    # People close to each other
    close_p = int(len(close_persons)//2)

    # to draw Count of close persons in Frame
    show_close_persons(image_np, close_p, im_height)

    # to draw line connecting the close persons
    for p1, p2 in close_persons:
        plot_close_lines(p1, p2, image_np)



