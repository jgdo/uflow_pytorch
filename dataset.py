import pickle
from torch.utils.data import DataLoader, TensorDataset
import uflow_utils
import torch

import cv2
import numpy as np
import random

def create_minecraft_loader(training, batch_size=64, shuffle=True, use_camera_actions=False):
    p = pickle.load(open('dataset/UFlow_data/ep1_pickle_doc.pkl', 'rb'))

    trainratio = 0.8
    train_len = int(len(p)*trainratio)
    if training:
        p = p[0:train_len]
    else:
        p = p[train_len:]

    img1 = []
    img2 = []
    actions = []

    for i in range(1, len(p)):
        img1.append(torch.from_numpy(p[i-1][0]).permute(2, 0, 1) / 255.0)
        img2.append(torch.from_numpy(p[i][0]).permute(2, 0, 1) / 255.0)
        cam_actions = torch.FloatTensor(p[i-1][1]['camera'] / 10.0) if use_camera_actions else torch.tensor([0.0, 0.0])
        actions.append(cam_actions)

    print('Loaded {} image pairs'.format(len(img1)))

    img1 = uflow_utils.upsample(torch.stack(img1).cuda(), is_flow=False, scale_factor=1)
    img2 = uflow_utils.upsample(torch.stack(img2).cuda(), is_flow=False, scale_factor=1)
    actions = torch.stack(actions).cuda()

    if False and training:
        img1 = uflow_utils.upsample(img1, is_flow=False, scale_factor=0.5)
        img2 = uflow_utils.upsample(img2, is_flow=False, scale_factor=0.5)
        img1 = uflow_utils.upsample(img1, is_flow=False, scale_factor=2)
        img2 = uflow_utils.upsample(img2, is_flow=False, scale_factor=2)

    # img1 = img1 * 2 - 1
    # img2 = img2 * 2 - 1

    dataset = TensorDataset(img1, img2, actions)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


def set_frame_point(frame, pos, type):
    try:
        if type:
            y,x = tuple(np.round(pos).astype(int))
            frame[y-2:y+3, x-2:x+3] = 1
            frame[y+1, x-2:x+1] = 0
            frame[y - 1, x - 2:x + 1] = 0

        else:
            frame[tuple(np.round(pos).astype(int))] = 1
            #frame[:] = cv2.GaussianBlur(frame, (5, 5), 0)
            # frame[:] *= 1.0 / frame[:].max()

            frame[:] = cv2.dilate(frame, np.ones((3,3)), iterations=1)

    except IndexError:
        pass

def generate_frame_seq(seq_len, H, W):
    type = random.choice([True, False])
    seq = np.zeros((seq_len, H, 64), dtype=np.float32)
    pose = [random.randint(5,H-6), random.randint(45, 55)]

    for i in range(seq_len):
        pose = [random.randint(7, H-8), random.randint(5, 59)]
        set_frame_point(seq[i], pose, type)
        # pose[1] += -3

    return seq

def generate_v(t, do_train, S):
    if t >= generate_moving_seq.num_objects:
        v = 0
    elif do_train:
        v = random.randrange(-4, 5)
        # v = generate_moving_seq.v
    else:
        v = generate_moving_seq.v
        generate_moving_seq.v += 1
        print('v = {}'.format(v))

    if v > 0:
        r = (5, 15)
    elif v < 0:
        r = (S - 15, S - 5)
    else:
        r = (5, S - 5)

    return v, r

def generate_moving_seq(seq_len, H, W, do_train):
    generate_moving_seq.num_total_objects = 1

    if do_train:
        bg_type = random.choice([True, False])
    else:
        bg_type = (generate_moving_seq.v & 2 == 0)

    orig_seq = None

    for t in range(generate_moving_seq.num_total_objects):
        this_seq = np.zeros((seq_len, H, W), dtype=np.float32)

        if do_train:
            type = random.choice([True, False])
        else:
            type = (generate_moving_seq.v & 1 == 0)

        v_x, r_x = generate_v(t, do_train, W)
        if True:
            v_y, r_y = generate_v(t, do_train, H)
        else:
            v_y = 0
            r_y = (5, H-5)

        pose = [random.randrange(*r_y), random.randrange(*r_x)]

        for i in range(seq_len):
            set_frame_point(this_seq[i], pose, type)
            pose[0] += v_y
            pose[1] += v_x

        if orig_seq is None:
            orig_seq = this_seq
        else:
            orig_seq = (orig_seq + this_seq).clip(0, 1)

    #if v < 0:
    #    seq = np.flip(seq, 2)

    background = np.zeros_like(orig_seq) + 0.1
    for bg in background:
        bg[5::(5 if bg_type else 10), :] = 0.5
        bg[:, 5::(10 if bg_type else 20)] = 0.5

    return orig_seq, background, None

generate_moving_seq.v = 0
generate_moving_seq.num_objects = 1

def gen_seq(seq_len, batch_size, H, W, do_train):
    obj = np.zeros((seq_len, batch_size, 1, H, W), dtype=np.float32)
    bg = np.zeros_like(obj)
    v = np.zeros((batch_size), dtype=np.float32)
    for b in range(batch_size):
        obj[:, b, 0], bg[:, b, 0], v[b] = generate_moving_seq(seq_len, H, W, do_train)

    obj = torch.from_numpy(obj).cuda()
    bg = torch.from_numpy(bg).cuda()
    combined = torch.clamp(obj + bg, 0, 1)
    return bg, obj, combined, v


def get_simple_moving_object_dataset(batch_size=64):
    seq_len = 15
    num_seq = 64
    _, _, data, _ = gen_seq(seq_len, num_seq, 64, 64, True)

    img1 = []
    img2 = []

    for seq_i in range(num_seq):
        for frame_i in range(1, seq_len):
            img1.append(data[frame_i-1, seq_i])
            img2.append(data[frame_i, seq_i])

    img1 = torch.stack(img1).cuda()
    img2 = torch.stack(img2).cuda()

    dataset = TensorDataset(img1, img2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader