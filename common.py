import cv2
import dlib
import numpy as np
from scipy import interpolate


def landmarks_to_img(landmarks, img_shape):
    landmarks *= [img_shape[1], img_shape[0]]
    canvas = np.zeros(img_shape, dtype=np.float32)
    draw_line_segments(canvas, landmarks, interpolation=False)

    return normalize(canvas)


def normalize(img):
    rank = len(img.shape)
    height_dim = 1 if rank == 4 else 0
    nearest_multiple_16 = img.shape[height_dim] // 16 * 16
    if nearest_multiple_16 != img.shape[height_dim]:
        # crop by height
        crop_need = img.shape[height_dim] - nearest_multiple_16
        if rank == 4:
            img = img[:, crop_need // 2:-crop_need // 2, :, :]
        else:
            img = img[crop_need // 2:-crop_need // 2, :, :]

    return img.astype(np.float32) / 255.0


def draw_points(img, points, color=(0, 0, 250)):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 3, color, cv2.FILLED, cv2.LINE_AA)


def draw_line_segments(img, points, interpolation=False, color=None):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (127, 0, 0),
        (0, 127, 0),
        (0, 0, 127),
        (127, 127, 0)
    ]
    points = points.astype(np.int32)
    chin = points[0:17]
    left_brow = points[17:22]
    right_brow = points[22:27]
    nose1 = points[27:31]
    nose1 = np.concatenate((nose1, [points[33]]))
    nose2 = points[31:36]
    left_eye = points[36:42]
    left_eye = np.concatenate((left_eye, [points[36]]))
    right_eye = points[42:48]
    right_eye = np.concatenate((right_eye, [points[42]]))
    mouth = points[48:60]
    mouth = np.concatenate((mouth, [points[48]]))
    mouth_internal = points[60:68]
    mouth_internal = np.concatenate((mouth_internal, [points[60]]))
    lines = np.array([
        chin, left_brow, right_brow,
        nose1, nose2, left_eye,
        right_eye, mouth, mouth_internal
    ])
    for i, line in enumerate(lines):
        if interpolation:
            tck, u = interpolate.splprep(line.transpose(), u=None, s=0.0, per=0)
            u_new = np.linspace(u.min(), u.max(), 100)
            xs, ys = interpolate.splev(u_new, tck, der=0)
            line = np.stack([xs, ys]).transpose()
        cur_color = colors[i]
        if color:
            cur_color = color
        cv2.polylines(
            img,
            np.int32([line]), False,
            cur_color, thickness=2, lineType=cv2.LINE_AA
        )


def get_boxes(face_driver, frame, threshold=0.5, offset=(0, 0)):
    input_name, input_shape = list(face_driver.inputs.items())[0]
    output_name = list(face_driver.outputs)[0]
    inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = face_driver.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    xmin = boxes[:, 0] * frame.shape[1] + offset[0]
    xmax = boxes[:, 2] * frame.shape[1] + offset[0]
    ymin = boxes[:, 1] * frame.shape[0] + offset[1]
    ymax = boxes[:, 3] * frame.shape[0] + offset[1]
    xmin[xmin < 0] = 0
    xmax[xmax > frame.shape[1]] = frame.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > frame.shape[0]] = frame.shape[0]

    boxes[:, 0] = xmin
    boxes[:, 2] = xmax
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0):
    h = (box[3] - box[1])
    w = (box[2] - box[0])
    ymin = int(max([box[1] - h * margin, 0]))
    ymax = int(min([box[3] + h * margin, img.shape[0]]))
    xmin = int(max([box[0] - w * margin, 0]))
    xmax = int(min([box[2] + w * margin, img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_landmarks(model, frame, box):
    box = box.astype(np.int)
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    shape = model(frame, rect)
    shape = shape_to_np(shape)

    return shape
