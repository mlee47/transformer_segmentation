
import os



# 파일 (ct경로,pt경로,gtvt경로)의 튜플로 리스트에 append
def get_paths_to_patient_files(path_to_imgs, append_mask=True):
    patients = os.listdir(path_to_imgs)
    patient_num = int(len(patients) / 3)
    paths = []

    for i in range(patient_num):
        a = 3 * i
        path_to_ct = path_to_imgs + '/' + str(patients[a])
        path_to_pt = path_to_imgs + '/' + str(patients[a + 2])

        if append_mask:
            path_to_mask = path_to_imgs + '/' + str(patients[a + 1])
            paths.append((path_to_ct, path_to_pt, path_to_mask))
        else:
            paths.append((path_to_ct, path_to_pt))

    return paths