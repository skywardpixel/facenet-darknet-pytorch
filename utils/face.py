import pickle
import os


def load_embeddings():
    with open("data/name", "r") as f:
        names = f.readlines()
    print(names)
    all_embeddings = {}
    for name in names:
        with open("model/" + name, "wb") as f:
            embedding = pickle.load(f)
            all_embeddings[name] = embedding
    return all_embeddings


def save_embeddings(username, data):
    with open("model/" + username, "wb") as f:
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
    os.mkdir("data/" + name, mode=755)
    return users
