import cv2
import cvui
import dlib
import numpy as np
import torch

from models import Darknet
from utils.face import load_embeddings, add_new_user, save_embeddings, NUM_EMB_EACH_USER, run_embeddings_knn
from utils.utils import static_vars


@static_vars(frame_idx=0, save_pic_idx=0)
def add_new_pic(frame, face_num, shape,
                object_pts, cam_matrix, dist_coeffs, users,
                euler_angle, last_euler_angle, warped_gray):
    SKIP_FRAME_COLLECTION = 10
    ANGLE_DIFF_TH = 5.0
    render_color = 255, 100, 100
    frame = cv2.rectangle(frame,
                          (frame.shape[1] // 2 - 150, frame.shape[0] // 2 - 150),
                          (300, 300),
                          color=render_color,
                          thickness=2)

    if face_num == 1:  # register one person each time
        cv2.putText(frame, "Please roll you head slightly", (30, 450), 0, 1.0, (255, 100, 100), 2)
        add_new_pic.frame_idx += 1

        left_brow_left_corner = (shape.part(17).x, shape.part(17).y)
        left_brow_right_corner = (shape.part(21).x, shape.part(21).y)
        right_brow_left_corner = (shape.part(22).x, shape.part(22).y)
        right_brow_right_corner = (shape.part(26).x, shape.part(26).y)
        left_eye_left_corner = (shape.part(36).x, shape.part(36).y)
        left_eye_right_corner = (shape.part(39).x, shape.part(39).y)
        right_eye_left_corner = (shape.part(42).x, shape.part(42).y)
        right_eye_right_corner = (shape.part(45).x, shape.part(45).y)
        nose_left_corner = (shape.part(31).x, shape.part(31).y)
        nose_right_corner = (shape.part(35).x, shape.part(35).y)
        mouth_left_corner = (shape.part(48).x, shape.part(48).y)
        mouth_right_corner = (shape.part(54).x, shape.part(54).y)
        mouth_bottom_corner = (shape.part(57).x, shape.part(57).y)
        chin_corner = (shape.part(8).x, shape.part(8).y)

        cv2.circle(frame, left_brow_left_corner, 3, render_color, -1)
        cv2.circle(frame, left_brow_right_corner, 3, render_color, -1)
        cv2.circle(frame, right_brow_left_corner, 3, render_color, -1)
        cv2.circle(frame, right_brow_right_corner, 3, render_color, -1)
        cv2.circle(frame, left_eye_left_corner, 3, render_color, -1)
        cv2.circle(frame, left_eye_right_corner, 3, render_color, -1)
        cv2.circle(frame, right_eye_left_corner, 3, render_color, -1)
        cv2.circle(frame, right_eye_right_corner, 3, render_color, -1)
        cv2.circle(frame, nose_left_corner, 3, render_color, -1)
        cv2.circle(frame, nose_right_corner, 3, render_color, -1)
        cv2.circle(frame, mouth_left_corner, 3, render_color, -1)
        cv2.circle(frame, mouth_right_corner, 3, render_color, -1)
        cv2.circle(frame, mouth_bottom_corner, 3, render_color, -1)
        cv2.circle(frame, chin_corner, 3, render_color, -1)

        if add_new_pic.frame_idx == SKIP_FRAME_COLLECTION:
            image_pts = np.array([
                left_brow_left_corner,
                left_brow_right_corner,
                right_brow_left_corner,
                right_brow_right_corner,
                left_eye_left_corner,
                left_eye_right_corner,
                right_eye_left_corner,
                right_eye_right_corner,
                nose_left_corner,
                nose_right_corner,
                mouth_left_corner,
                mouth_right_corner,
                mouth_bottom_corner,
                chin_corner,
            ], dtype=np.float64)

            # calc pose
            ret_val, rotation_vec, translation_vec = cv2.solvePnP(object_pts,
                                                                  image_pts,
                                                                  cam_matrix,
                                                                  dist_coeffs)

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            out_intrinsics, out_rotation, out_translation, _, _, _, euler_angle \
                = cv2.decomposeProjectionMatrix(pose_mat)

            x_angle = euler_angle[0] - last_euler_angle[0]
            y_angle = euler_angle[1] - last_euler_angle[1]
            z_angle = euler_angle[2] - last_euler_angle[2]
            # save user pic in different angle
            if abs(x_angle) > ANGLE_DIFF_TH or abs(y_angle) > ANGLE_DIFF_TH or abs(z_angle) > ANGLE_DIFF_TH:
                username = users[-1]
                pic_file = "data/{}/{}.jpg".format(username, add_new_pic.save_pic_idx)
                cv2.imwrite(pic_file, warped_gray)
                last_euler_angle = [
                    euler_angle[0],
                    euler_angle[1],
                    euler_angle[2],
                ]
                add_new_pic.save_pic_idx += 1
            add_new_pic.frame_idx = 0

    if add_new_pic.save_pic_idx == NUM_EMB_EACH_USER:
        add_new_pic.save_pic_idx = 0
        return True, euler_angle, last_euler_angle
    else:
        return False, euler_angle, last_euler_angle


def run_embedding(model, aligned_face):
    img = np.array(aligned_face)
    # h, w = img.shape
    # Resize and normalize
    # input_img = skimage.transform.resize(img, (160, 3), mode='reflect')
    # Channels-first
    input_img = np.repeat(img[np.newaxis, :, :], 3, axis=0)
    input_img = np.expand_dims(input_img, 0)
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()

    embedding = model(input_img)
    torch.norm(embedding).detach()
    return embedding


def produce_features(model, username):
    to_be_saved_embeddings = []
    for i in range(NUM_EMB_EACH_USER):
        pic_file = "data/{}/{}.jpg".format(username, i)
        temp = cv2.imread(pic_file, 0)
        embedding = run_embedding(model, temp)
        to_be_saved_embeddings.append(embedding)
    save_embeddings(username, to_be_saved_embeddings)
    return to_be_saved_embeddings


def main():
    face_cascade = cv2.CascadeClassifier("weights/haarcascade_frontalface_alt2.xml")
    predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")
    embeddings = load_embeddings()
    if embeddings:
        users, embeddings = zip(*embeddings)
        users = list(users)
        embeddings = list(embeddings)
    else:
        users, embeddings = [], []
    model = Darknet("config/facenet.cfg", img_size=160)
    model.load_weights("weights/facenet.weights")

    warp_scale = 1.0
    front_face_pts = [
        (58.20558929, 28.47149849),
        (99.03411102, 27.64450073),
        (80.03263855, 120.09350586),
    ]
    front_face_pts = np.array([(warp_scale * a, warp_scale * b)
                               for a, b in front_face_pts],
                              dtype=np.float64)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to access video 0")
        exit(1)

    _, frame = cap.read()

    c_x = frame.shape[1] / 2.0
    c_y = frame.shape[1] / 2.0
    f_x = c_x * 1.732050808
    # K = [f_x, 0.0, c_x, 0.0, f_x, c_y, 0.0, 0.0, 1.0]
    # D = [0.0, 0.0, 0.0, 0.0, 0.0]

    cam_matrix = np.array(
        [[f_x, 0.0, c_x],
         [0.0, f_x, c_y],
         [0.0, 0.0, 1.0]],
        dtype=np.float64
    )

    dist_coeffs = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64
    )

    object_pts = np.array([
        (6.825897, 6.760612, 4.402142),  # 33 left brow left corner
        (1.330353, 7.122144, 6.903745),  # 29 left brow right corner
        (-1.330353, 7.122144, 6.903745),  # 34 right brow left corner
        (-6.825897, 6.760612, 4.402142),  # 38 right brow right corner
        (5.311432, 5.485328, 3.987654),  # 13 left eye left corner
        (1.789930, 5.393625, 4.413414),  # 17 left eye right corner
        (-1.789930, 5.393625, 4.413414),  # 25 right eye left corner
        (-5.311432, 5.485328, 3.987654),  # 21 right eye right corner
        (2.005628, 1.409845, 6.165652),  # 55 nose left corner
        (-2.005628, 1.409845, 6.165652),  # 49 nose right corner
        (2.774015, -2.080775, 5.048531),  # 43 mouth left corner
        (-2.774015, -2.080775, 5.048531),  # 39 mouth right corner
        (0.000000, -3.116408, 6.097667),  # 45 mouth central bottom corner
        (0.000000, -7.415691, 4.070434),  # 6 chin corner
    ], dtype=np.float64)

    # pose_mat = np.zeros((3, 4, 1), dtype=np.float64)
    euler_angle = np.zeros((3, 1, 1), dtype=np.float64)
    last_euler_angle = euler_angle.copy()
    # out_intrinsics = np.zeros((3, 3, 1), dtype=np.float64)
    # out_rotation = np.zeros((3, 3, 1), dtype=np.float64)
    # out_translation = np.zeros((3, 1, 1), dtype=np.float64)

    # cvui.init("DarkFace")

    # states
    is_registering, is_recognizing, is_putting_text, is_adding_name = False, False, False, False
    put_text_countdown = 0
    text_to_put = ''
    global_color = 255, 255, 255

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        #                               # cv2.CV_HAAR_SCALE_IMAGE | cv2.CV_HAAR_FIND_BIGGEST_OBJECT,
        #                               # (160, 160),
        #                               # (400, 400))
        face_num = len(faces)
        if face_num > 0:
            dkrect = dlib.rectangle(faces[0][0], faces[0][1], faces[0][0] + faces[0][2], faces[0][1] + faces[0][3])
            shape = predictor(gray, dkrect)
            current_face_pts = [
                (shape.part(39).x, shape.part(39).y),
                (shape.part(42).x, shape.part(42).y),
                (shape.part(57).x, shape.part(57).y)
            ]
            np_current_face_pts = np.float32(current_face_pts)
            np_front_face_pts = np.float32(front_face_pts)
            to_front_H = cv2.getAffineTransform(np_current_face_pts, np_front_face_pts)
            warped_gray = cv2.warpAffine(gray, to_front_H, (160, 160))

        if is_recognizing:
            if face_num > 0:
                bl = faces[0][0], faces[0][1] + faces[0][3]
                br = faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]
                frame = cv2.line(frame, pt1=bl, pt2=br, color=(255, 0, 0), thickness=2)
            embedding = run_embedding(model, warped_gray)
            user_idx, confidence = run_embeddings_knn(embedding, users, embeddings)
            name = users[user_idx] if user_idx < len(users) else "unknown"
            print("recognized " + name)

        elif is_registering:
            if is_adding_name:
                is_add = add_new_user(users)
                if not is_add:
                    is_registering = False
                    is_putting_text = True
                    text_to_put = "User name already exists"
                    global_color = 50, 50, 255
                    continue
                is_adding_name = False

            added, euler_angle, last_euler_angle \
                = add_new_pic(frame,
                              face_num=face_num,
                              shape=shape,
                              object_pts=object_pts,
                              cam_matrix=cam_matrix,
                              dist_coeffs=dist_coeffs,
                              users=users,
                              euler_angle=euler_angle,
                              last_euler_angle=last_euler_angle,
                              warped_gray=warped_gray
                              )
            if added:
                new_embeddings = produce_features(model, users[-1])
                embeddings.extend([(users[-1], e) for e in new_embeddings])
                is_putting_text = True
                text_to_put = "Registration complete"
                global_color = 50, 255, 50
                is_registering = False
                cv2.imshow('frame', frame)

        if is_putting_text:
            if put_text_countdown < 30:
                if face_num:
                    cv2.putText(frame, text_to_put, (30, 450), 0, 1.0, global_color, 2)
            else:
                put_text_countdown = 0
                is_putting_text = False

        # # display buttons
        # if cvui.button(frame, 0, 0, 100, 30, "&Add user"):
        #     is_recognizing = 0
        #     is_registering = 1
        #     is_adding_name = 1
        #     print("Register mode!")
        #
        # if cvui.button(frame, 100, 0, 200, 30, "&Start / stop recognition"):
        #     is_recognizing = not is_recognizing
        #     print("Recognizing..." if is_recognizing else "Not recognizing.")
        #
        # if cvui.button(frame, 300, 0, 75, 30, "&Quit"):
        #     print("Exiting")
        #     break
        #
        # cvui.update()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(100)
        if key == ord('q'):
            print("Exiting")
            break
        elif key == ord('a'):
            is_recognizing = 0
            is_registering = 1
            is_adding_name = 1
            print("Register mode!")
        elif key == ord('r'):
            is_recognizing = not is_recognizing
            print("Recognizing..." if is_recognizing else "Not recognizing.")


if __name__ == '__main__':
    main()
