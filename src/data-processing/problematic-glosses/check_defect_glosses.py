import json

# read gloss data
with open('gloss_dictionary.json') as lf:
    gloss_data: dict = json.load(lf)

# write all findings into txt
with open('debugging_log.txt', 'a') as wf:

    keypoints_meta: list[tuple[int, str]] = [
        (39, 'pose_keypoints_2d'),
        (210, 'face_keypoints_2d'),
        (63, 'hand_left_keypoints_2d'),
        (63, 'hand_right_keypoints_2d')
    ]

    # track glosses that have not expected amount of keypoint elements
    # (set assures only uniques, without same frame)
    list_of_defect_glosses: set[tuple] = set()

    for expected_num_of_list_elements, body_part in keypoints_meta:

        for gloss_key, gloss_keypoints in gloss_data.items():
            for i_frame, gloss_frame in enumerate(gloss_keypoints):

                num_of_list_elements: int = len(gloss_frame[body_part])

                if num_of_list_elements != expected_num_of_list_elements:
                    wf.write('------------\n')
                    wf.write(f'Gloss: {gloss_key}\n')
                    wf.write(f'Body part: {body_part}\n')
                    wf.write(f'Frame: {i_frame}\n')
                    wf.write(f'Expected number of list elements: {expected_num_of_list_elements}\n')
                    wf.write(f'Actual number of list elements: {num_of_list_elements}\n')
                    wf.write('\n')

                    list_of_defect_glosses.add(
                        (gloss_key, body_part, expected_num_of_list_elements, num_of_list_elements)
                    )

    wf.write(f'\nDEFECT GLOSSES ({len(list_of_defect_glosses)})\n')
    wf.write('GLOSS | BODY PART | EXPECTED | ACTUAL\n')
    for defect_gloss in list_of_defect_glosses:
        wf.write(f'{defect_gloss}\n')