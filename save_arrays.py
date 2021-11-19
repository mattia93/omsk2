import pickle


def save(object, name):
    file = open(name, "wb")
    pickle.dump(object, file)
    file.close()