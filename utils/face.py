import pickle
import os


def load_embeddings():
    with open("data/name", "r") as f:
        names = f.readlines()
    print(names)
    embeddings = []
    for name in names:
        with open("model/" + name, "rb") as f:
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
