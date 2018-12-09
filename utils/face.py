import pickle
import os

import torch

NUM_EMB_EACH_USER = 3
KNN_NUM = 10
DISTANCE_THRESHOLD = 1.0
MAX_NUM_USER = 100


def load_embeddings():
    with open("data/name", "r") as f:
        names = f.readlines()
    print(names)
    embeddings = []
    for name in names:
        with open("model/{}.pkl".format(name.strip()), "rb") as f:
            embedding = pickle.load(f)
            embeddings.append((name, embedding))
    return embeddings


def save_embeddings(username, data):
    if not os.path.exists("model"):
        os.mkdir("model", 0o755)
    with open("model/{}.pkl".format(username), "wb+") as f:
        pickle.dump(data, f)


def add_new_user(users):
    name = input("Type new user's name: ").strip()
    while name in users:
        print('User name already exists!')
        return False

    with open("data/name", "a") as f:
        f.write(name + '\n')
        f.close()
    users.append(name)
    os.mkdir("data/" + name, mode=0o755)
    return True


def run_embeddings_knn(src, users, embeddings):
    print(len(users))
    num_comparison = len(users) * NUM_EMB_EACH_USER

    # saving (label, distance)
    all_id_and_dist = []

    candidates = []

    for i in range(len(users)):
        usercase = embeddings[i]
        for j in range(NUM_EMB_EACH_USER):
            # Euclidean distance
            distance = torch.dist(src, usercase[j]).detach()
            all_id_and_dist.append((i, distance))

    print(all_id_and_dist)

    # sort by least distance
    all_id_and_dist.sort(key=(lambda tup: tup[1]))

    if num_comparison > KNN_NUM:
        num_comparison = KNN_NUM

    for i in range(num_comparison):
        id, dist = all_id_and_dist[i]
        if dist < DISTANCE_THRESHOLD:
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
