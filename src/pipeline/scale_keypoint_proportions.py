import os
import json
import glob

def get_bbox_dims(keypoints):
    if not keypoints:
        return 0.0, 0.0
    xs = keypoints[0::3]
    ys = keypoints[1::3]
    return max(xs) - min(xs), max(ys) - min(ys)


def get_bbox_center(keypoints):
    if not keypoints:
        return 0.0, 0.0
    xs = keypoints[0::3]
    ys = keypoints[1::3]
    return (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def compute_default_body_dims(first_json_path):
    data = load_json(first_json_path)
    return get_bbox_dims(data['pose_sequence'][0]['pose_keypoints_2d'])


def compute_default_face_dims_and_center(first_json_path):
    data = load_json(first_json_path)
    kpts = data['pose_sequence'][0].get('face_keypoints_2d', [])
    return (*get_bbox_dims(kpts), *get_bbox_center(kpts))


def scale_frame_array(kpts, sx, sy):
    for i in range(0, len(kpts), 3):
        kpts[i]   *= sx
        kpts[i+1] *= sy


def scale_all_files(folder_path, uniform=True):
    files = sorted(glob.glob(os.path.join(folder_path, '*.json')),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    if not files:
        raise RuntimeError(f"No JSON files found in {folder_path}")

    # 1.json defines our “golden” body & face
    default_body_w, default_body_h = compute_default_body_dims(files[0])
    default_face_w, default_face_h, default_face_cx, default_face_cy = \
        compute_default_face_dims_and_center(files[0])

    print(f"Default body dims: {default_body_w:.1f}×{default_body_h:.1f}")
    print(f"Default face dims: {default_face_w:.1f}×{default_face_h:.1f}, center @ ({default_face_cx:.1f},{default_face_cy:.1f})")

    for path in files[1:]:
        data = load_json(path)
        first = data['pose_sequence'][0]

        # --- 1) compute body scale from first frame
        bw, bh = get_bbox_dims(first['pose_keypoints_2d'])
        if bw == 0 or bh == 0:
            print(f"Skipping {os.path.basename(path)} (zero-body bbox)")
            continue
        sx_body = default_body_w / bw
        sy_body = default_body_h / bh
        if uniform:
            s = (sx_body + sy_body) / 2.0
            sx_body = sy_body = s

        # --- 2) compute face scale from first frame
        face0 = first.get('face_keypoints_2d', [])
        fw, fh = get_bbox_dims(face0)
        if fw > 0 and fh > 0:
            sx_face = default_face_w / fw
            sy_face = default_face_h / fh
            if uniform:
                s = (sx_face + sy_face) / 2.0
                sx_face = sy_face = s
        else:
            sx_face = sy_face = None

        print(f"Scaling {os.path.basename(path)}: body→({sx_body:.3f},{sy_body:.3f}), face→({sx_face},{sy_face})")

        # --- 3) apply to every frame
        for frame in data['pose_sequence']:
            # scale body & hands
            for key in ('pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d'):
                arr = frame.get(key, [])
                scale_frame_array(arr, sx_body, sy_body)
                frame[key] = arr

            # scale & re-attach face
            if sx_face is not None:
                fk = frame['face_keypoints_2d']
                # a) find raw center
                cx, cy = get_bbox_center(fk)
                # b) shift to origin
                for i in range(0, len(fk), 3):
                    fk[i]   -= cx
                    fk[i+1] -= cy
                # c) resize around origin
                for i in range(0, len(fk), 3):
                    fk[i]   *= sx_face
                    fk[i+1] *= sy_face
                # d) translate to where the head _should_ be after body-scaling
                tx, ty = cx * sx_body, cy * sy_body
                for i in range(0, len(fk), 3):
                    fk[i]   += tx
                    fk[i+1] += ty

                frame['face_keypoints_2d'] = fk

        save_json(data, path)

    print("Scaling & re-centering complete.")

if __name__ == "__main__":
    scale_all_files("gloss2pose_data")
