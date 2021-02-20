import pickle

def load(filename): 
  with open(filename, 'rb') as f: 
    return pickle.load(f)

def save(filename, object):
  with open(filename, 'wb') as f: 
    return pickle.dump(object, f)