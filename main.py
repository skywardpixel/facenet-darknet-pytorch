import cv2
import dlib
import cvui
import numpy as np

from models import Darknet


def main():
    face_cascade = cv2.CascadeClassifier("weights/haarcascade_frontalface_alt2.xml")
    predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")
    # load_embeddings("data/names")
    model = Darknet("config/facenet.cfg")


    warp_scale = 1.0
    front_face_pts = [
        (58.20558929, 28.47149849),
        (99.03411102, 27.64450073),
        (80.03263855, 120.09350586),
    ]
    front_face_pts = [(warp_scale * a, warp_scale * b)
                      for a, b in front_face_pts]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to access video 0")
        exit(1)

    _, frame = cap.read()
    print(frame.shape)
    c_x = frame.shape[1] / 2.0
    c_y = frame.shape[1] / 2.0
    f_x = c_x * 1.732050808
    K = [f_x, 0.0, c_x, 0.0, f_x, c_y, 0.0, 0.0, 1.0]
    D = [0.0, 0.0, 0.0, 0.0, 0.0]

    cam_matrix = np.array(
        [[f_x, 0.0, c_x],
         [0.0, f_x, c_y],
         [0.0, 0.0, 1.0]],
        dtype="float64"
    )
    dist_coeffs = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        dtype="float64"
    )

    object_pts = [
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
    ]

    pose_mat = np.zeros((3, 4, 1), dtype="float64")
    euler_angle = np.zeros((3, 1, 1), dtype="float64")
    out_intrinsics = np.zeros((3, 3, 1), dtype="float64")
    out_rotation = np.zeros((3, 3, 1), dtype="float64")
    out_translation = np.zeros((3, 1, 1), dtype="float64")

    cvui.init("DarkFace")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', gray)
        # dkgray = dlib.cv_image(gray)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        #                               # cv2.CV_HAAR_SCALE_IMAGE | cv2.CV_HAAR_FIND_BIGGEST_OBJECT,
        #                               # (160, 160),
        #                               # (400, 400))
        print(faces)
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

            if face_num > 0:
                cv2.line(frame,
                         (faces[0][0], faces[0][1] + faces[0][3]),
                         (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]),
                         (255, 0, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) == ord('q'):
            break


if __name__ == '__main__':
    main()
