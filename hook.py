import logging

import cv2
import face_alignment
import numpy as np
from ml_serving.utils import helpers
import torch

import common
from dataset import video_extraction_conversion
from network import model


PARAMS = {
    'target': '',
    'torch_model': '',
    'margin': 0.4,
    'image_size': (256, 256),
}


LOG = logging.getLogger(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')


def init_hook(ctx, **params):
    PARAMS.update(params)
    PARAMS['margin'] = float(PARAMS['margin'])

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device.type)
    ref_img = cv2.imread(PARAMS['target'])
    # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    # find face on image
    face_driver = ctx.drivers[0]
    boxes = common.get_boxes(face_driver, ref_img)
    if len(boxes) != 1:
        raise RuntimeError('target image must include exactly 1 face. Provide path via -o target=<path>')

    face = common.crop_by_box(ref_img, boxes[0], margin=PARAMS['margin'])
    PARAMS['face_shape'] = face.shape

    face = cv2.resize(face, (PARAMS['image_size'][1], PARAMS['image_size'][0]), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    norm_image = torch.from_numpy(np.expand_dims(face, axis=0)).type(dtype=torch.float)  # K,256,256,3
    norm_image = (norm_image.permute([0, 3, 1, 2]) - 127.5) / 127.5
    PARAMS['face'] = norm_image
    PARAMS['full'] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    LOG.info('Loading torch model...')
    torch_model = load_torch_model(PARAMS['torch_model'])
    LOG.info('Done loading torch model.')

    return fa, torch_model


def load_torch_model(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    net = torch.nn.DataParallel(model.Generator(PARAMS['image_size'][0], device).to(device))
    net.module.load_state_dict(checkpoint['state_dict'], strict=False)
    net.eval()

    return net


def get_picture(tensor):
    return (tensor[0] * 127.5 + 127.5).permute([1, 2, 0]).to(cpu).numpy().clip(0, 255).astype(np.uint8)


def process(inputs, ctx, **kwargs):
    image, is_video = helpers.load_image(inputs, 'input')
    fa, torch_model = ctx.global_ctx

    face_driver = ctx.drivers[0]

    boxes = common.get_boxes(face_driver, image)

    # TODO: for example
    output = np.zeros([256, 256, 3]).astype(np.uint8)

    for box in boxes:
        crop_box = common.get_crop_box(image, box, margin=PARAMS['margin'])
        cropped = common.crop_by_box(image, box, margin=PARAMS['margin'])
        resized = cv2.resize(cropped, (PARAMS['image_size'][1], PARAMS['image_size'][0]), interpolation=cv2.INTER_AREA)

        landmarks = fa.get_landmarks_from_image(image, [crop_box])
        landmark_img = video_extraction_conversion.draw_landmark(
            landmarks[0], size=(PARAMS['image_size'][0], PARAMS['image_size'][1], 3)
        )

        norm_image = torch.from_numpy(np.expand_dims(resized, axis=0)).type(dtype=torch.float)  # K,256,256,3
        norm_mark = torch.from_numpy(np.expand_dims(landmark_img, axis=0)).type(dtype=torch.float)  # K,256,256,3
        norm_image = (norm_image.permute([0, 3, 1, 2]) - 127.5) / 127.5
        norm_mark = (norm_mark.permute([0, 3, 1, 2]) - 127.5) / 127.5  # K,3,256,256

        cv2.imwrite('VV.jpg', get_picture(norm_mark))
        import sys
        sys.exit(0)
        outputs = torch_model(PARAMS['face'], norm_mark)



        output = get_picture(outputs)
        # output = cv2.resize(output, (PARAMS['face_shape'][1], PARAMS['face_shape'][0]), interpolation=cv2.INTER_AREA)

        # image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = output
        # mask = np.ones_like(output) * 255
        # center_box = (int((crop_box[0] + crop_box[2]) // 2), int((crop_box[1] + crop_box[3]) // 2))
        # image = cv2.seamlessClone(output, image, mask, center_box, cv2.NORMAL_CLONE)

    if is_video:
        output = output
    else:
        _, buf = cv2.imencode('.jpg', output[:, :, ::-1])
        output = buf.tostring()

    return {'output': output}
