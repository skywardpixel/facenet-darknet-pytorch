import pickle
import os

import torch

def load_embeddings(opt):
    with open(opt.names_path, "r") as f:
        names = f.readlines()
    print(names)
    embeddings = []
    for name in names:
        with open(os.path.join(opt.embeddings_path, name + ".pkl"), "rb") as f:
            embedding = pickle.load(f)
            embeddings.append((name, embedding))
    return embeddings


def save_embeddings(username, data, opt):
    if not os.path.exists(opt.embeddings_path):
        os.mkdir(opt.embeddings_path, 0o755)
    with open(os.path.join(opt.embeddings_path, username + ".pkl"), "wb+") as f:
        pickle.dump(data, f)


def add_new_user(names_path, users):
    name = input("Type new user's name: ").strip()
    while name in users:
        print('User name already exists!')
        return False

    with open(names_path, "a") as f:
        f.write(name + '\n')
        f.close()
    users.append(name)
    os.mkdir(os.path.join("data", name), mode=0o755)
    return True


def run_embeddings_knn(src, users, embeddings, opt):
    MAX_NUM_USER = 100
    num_comparison = len(users) * opt.num_embeddings

    # saving (label, distance)
    all_id_and_dist = []

    candidates = []

    for i in range(len(users)):
        usercase = embeddings[i]
        for j in range(opt.num_embeddings):
            # Euclidean distance
            distance = torch.dist(src, usercase[j]).detach()
            all_id_and_dist.append((i, distance))

    print(all_id_and_dist)

    # sort by least distance
    all_id_and_dist.sort(key=(lambda tup: tup[1]))

    if num_comparison > opt.knn_num:
        num_comparison = opt.knn_num

    for i in range(num_comparison):
        id, dist = all_id_and_dist[i]
        if dist < opt.knn_dist_thres:
            candidates.append((id, dist))

    if not candidates:
        recog_name_idx = MAX_NUM_USER
        confidence = 2.0

    else:
        hitsum = [0] * MAX_NUM_USER
        for c in candidates:
            hitsum[c[0]] += 1

        max_hitsum = 0
        max_hitidx = 0
        min_dist = 2.0
        for i in range(MAX_NUM_USER):
            if max_hitsum < hitsum[i]:
                max_hitsum = hitsum[i]
                max_hitidx = i
        for c in candidates:
            if c[0] == max_hitidx:
                if min_dist > c[1]:
                    min_dist = c[1]

        recog_name_idx = max_hitidx
        confidence = min_dist

    return recog_name_idx, confidence
