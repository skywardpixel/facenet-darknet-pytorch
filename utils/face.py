import pickle
import os

import torch

NUM_EMB_EACH_USER = 3
KNN_NUM = 10
DISTANCE_THRESHOLD = 0.7
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
        name = input("Type new user's name: ")

    with open("data/name", "a") as f:
        f.write(name + '\n')
        f.close()
    users.append(name)
    os.mkdir("data/" + name, mode=0o755)
    return users


def run_embeddings_knn(src, users, embeddings):

    num_comparison = len(users) * NUM_EMB_EACH_USER

    # saving (label, distance)
    all_id_and_dist = []

    num_candidates = 0
    candidates = []

    for i in range(len(users)):
        usercase = embeddings[i]
        for j in range(NUM_EMB_EACH_USER):
            # Euclidean distance
            distance = torch.dist(src, usercase[j])
            all_id_and_dist.append((i, distance))

    # sort by least distance
    all_id_and_dist.sort(key=(lambda tup: tup[1]))

    if num_comparison > KNN_NUM:
        num_comparison = KNN_NUM

    for i in range(num_comparison):
        id, dist = all_id_and_dist[i]
        if dist < DISTANCE_THRESHOLD:
            candidates.append((id, dist))

    if num_candidates == 0:
        recog_name_idx = MAX_NUM_USER
        confidence = 2.0

    else:
        hitsum = [0] * MAX_NUM_USER
        for i in range(num_candidates):
            hitsum[candidates[i][0]] += 1

        max_hitsum = 0
        max_hitidx = 0
        min_dist = 2.0
        for i in range(MAX_NUM_USER):
            if max_hitsum < hitsum[i]:
                max_hitsum = hitsum[i]
                max_hitidx = i
        for i in range(num_candidates):
            idx = i * 2
            if candidates[idx + 1] == max_hitidx:
                if min_dist > candidates[idx]:
                    min_dist = candidates[idx]

        recog_name_idx = max_hitidx
        confidence = min_dist

    return recog_name_idx, confidence
