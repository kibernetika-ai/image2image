import cv2
import numpy as np
from ml_serving.utils import helpers

import common


PARAMS = {
    'landmark_model': '',
    'reference': '',
    'margin': 0.2,
}


def init_hook(ctx, **params):
    PARAMS.update(params)

    landmark_driver = common.load_shape_model(PARAMS['landmark_model'])
    ref_img = cv2.imread(PARAMS['reference'])
    # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    # find face on image
    face_driver = ctx.drivers[0]
    boxes = common.get_boxes(face_driver, ref_img)
    if len(boxes) != 1:
        raise RuntimeError('reference image must include exactly 1 face. Provide path via -o reference=<path>')

    face = common.crop_by_box(ref_img, boxes[0], margin=PARAMS['margin'])
    face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
    face = common.normalize(face)
    # insert RGB
    PARAMS['ref'] = face[:, :, ::-1]

    return landmark_driver


def process(inputs, ctx, **kwargs):
    image, is_video = helpers.load_image(inputs, 'input')

    face_driver = ctx.drivers[0]
    im2im_driver = ctx.drivers[1]
    inp_name1 = list(im2im_driver.inputs.keys())[0]
    inp_name2 = list(im2im_driver.inputs.keys())[1]

    boxes = common.get_boxes(face_driver, image)
    for box in boxes:
        crop_box = common.get_crop_box(image, box, margin=PARAMS['margin'])
        cropped = common.crop_by_box(image, box, margin=PARAMS['margin'])
        resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
        new_box = np.array([
            max(box[0] - crop_box[0], 0),
            max(box[1] - crop_box[1], 0),
            min(crop_box[2] - box[0], cropped.shape[1]),
            min(crop_box[3] - box[1], cropped.shape[0]),
        ])

        landmarks = common.get_landmarks(ctx.global_ctx, cropped, new_box)
        landmarks = landmarks.astype(np.float32) / [cropped.shape[1], cropped.shape[0]]
        landmark_img = common.landmarks_to_img(landmarks, resized.shape)
        # resized = common.normalize(resized)

        outputs = im2im_driver.predict({
            inp_name1: np.expand_dims(PARAMS['ref'], axis=0),
            inp_name2: np.expand_dims(landmark_img, axis=0)
        })
        output = list(outputs.values())[0].reshape(256, 256, 3)
        output = (output * 255.0).astype(np.uint8).clip(0, 255)
        output = cv2.resize(output, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_AREA)

        image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = output
        # cv2.seamlessClone(output, image, )

    if is_video:
        output = image
    else:
        _, buf = cv2.imencode('.jpg', image[:, :, ::-1])
        output = buf.tostring()

    return {'output': output}
