import numpy as np
import os,sys
import pickle

# load learned embeddings and dictionary to "words"
def load_learned_embeddings(embeddings_filepath, reverse_dictionary_filepath, dictionary_filepath, w2i_dictionary_filepath, w2c_dictionary_filepath):
    if(os.path.exists(embeddings_filepath)):
        print("embeddings loaded")
        embeddings = np.load(embeddings_filepath)
    if(os.path.exists(reverse_dictionary_filepath)):
        with open(reverse_dictionary_filepath, 'rb') as handle:
            print("reverse_dict loaded")
            reverse_dictionary = pickle.load(handle)
    if(os.path.exists(dictionary_filepath)):
        with open(dictionary_filepath, 'rb') as handle:
            print("dict loaded")
            dictionary = pickle.load(handle)
    if(os.path.exists(w2i_dictionary_filepath)):
        with open(w2i_dictionary_filepath, 'rb') as handle:
            print("w2i_dict loaded")
            w2i_dictionary = pickle.load(handle)
    if(os.path.exists(w2c_dictionary_filepath)):
        with open(w2c_dictionary_filepath, 'rb') as handle:
            print("w2c_dict loaded")
            w2c_dictionary = pickle.load(handle)
    return embeddings, reverse_dictionary, dictionary, w2i_dictionary, w2c_dictionary

# determine filepaths to load based on which model we want to look at
def create_filepaths(DATASET,SIZE,SIZE_WINDOW, QUANT_FACTOR):
    datapath = './data'
    emb_file = 'final_embeddings_Q{0}_sw{1}_e{2}_fr{3}.npy'.format(QUANT_FACTOR, SIZE_WINDOW,SIZE,DATASET);
    rev_dict_file = 'reverse_dictionary_fr{0}Q{1}.pkl'.format(DATASET, QUANT_FACTOR)
    dict_file = 'dictionary_fr{0}Q{1}.pkl'.format(DATASET, QUANT_FACTOR)
    w2i_file = 'w2i_dictionary_fr{0}Q{1}.pkl'.format(DATASET, QUANT_FACTOR)
    w2c_file = 'w2c_dictionary_fr{0}Q{1}.pkl'.format(DATASET, QUANT_FACTOR)

    embeddings_filepath = os.path.join(datapath,emb_file)
    print(embeddings_filepath)
    reverse_dictionary_filepath = os.path.join(datapath,rev_dict_file)
    print(reverse_dictionary_filepath)
    dictionary_filepath = os.path.join(datapath,dict_file)
    print(dictionary_filepath)
    w2i_dictionary_filepath = os.path.join(datapath,w2i_file)
    print(w2i_dictionary_filepath)
    w2c_dictionary_filepath = os.path.join(datapath,w2c_file)
    print(w2c_dictionary_filepath)

    return embeddings_filepath, reverse_dictionary_filepath, dictionary_filepath, w2i_dictionary_filepath, w2c_dictionary_filepath


def loadUMAP(filepath):
    return np.load(filepath)

# metadata parsing function for a single point
def parseMetadata(metadata,hlist):
    mlist = metadata.split("#") #known delimiter
    meta_dict = dict(zip(hlist, mlist))
    return meta_dict

def loadMetadata(filepath):
    with open(filepath) as f:
        metadata_all = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    metadata_all = [x.strip() for x in metadata_all]
    header = metadata_all[0]
    hlist = header.split("#")
    metadata_all = metadata_all[1:]
    return metadata_all, hlist

def loadSiameseEmbeddings(filepath, TR_SIZE=3000):
    npz = np.load(filepath)
    E = npz['Efull']
    F = npz['Ffull']
    #S = npz['Sfull']#currently missing
    S = []
    [ S.extend([100,1000,1500,2000,2500]) for i in range(TR_SIZE)]
    if(len(F) > E.shape[0]):
      F = F[:E.shape[0]]
    if(len(S) > E.shape[0]):
      S = S[:E.shape[0]]
    return E,F,S

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def loadNpyFileNames(filepath, MIN, MAX):
    fnames = read_data(filepath)
    if (len(fnames) > (MAX-MIN)):
        fnames = fnames[MIN:MAX]
    return fnames

# load .npy contour section
def LoadNpyPitchTrack(filepath, frame_start, frame_length = 350):
  pt = np.load(filepath)
  pt = pt[frame_start:frame_start+frame_length]
  pt = [(cval-pt[0]) for cval in pt] #set all contours to start at 0 cents
  return pt

#remove blank metadata indices
def removeEntriesWithNoMetadata(START,END,metadata,F,S,E,u,contours):
  metadata = metadata[:START] + metadata[END+1:]
  S = S[:START] + S[END+1:]
  blank = range(START,END+1)
  E = np.delete(E,blank,axis=0)
  F = np.delete(F,blank,axis=0)
  u = np.delete(u,blank,axis=0)
  contours = np.delete(contours,blank,axis=0)
  return metadata, F, S, E, u, contours
