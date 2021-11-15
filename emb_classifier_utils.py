#emb_classifier_utils.py
import os
import numpy as np
import math
from analysis_helpers import *

# Load from compute_embeddings output format, an npz file with E, F, SF and C
class EmbeddingLoader:
    def __init__(self):
        pass
    def filterNone(self,L):
        return [x for x in L if x is not None]

    def Load(self,relativeFilePath):
        NPZ = np.load(relativeFilePath, allow_pickle=True)
        E = NPZ['E']
        F = NPZ['F']
        SF = NPZ['SF']
        C = NPZ['C']
        print('embeddings loaded')

        E = self.filterNone(E)
        F = self.filterNone(F)
        SF = self.filterNone(SF)
        C = self.filterNone(C)

        print('Num samples:',len(E), 'Dimensions:', E[0].shape)
        return E,F,SF,C

#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create Corresponding Labels from VocalSet Filename Metadata
class VocalSetLabels:
    def __init__(self):
        pass
    def removeItem(self,item, labels):
        if item in labels: labels.remove(item)
        return labels

    def replaceItem(self,item1,item2,labels):
        if(item1 in labels):
            labels[labels.index(item1)] = item2
        return labels

    def combine(self,item1, item2, labels):
        if((item1 in labels) and (item2 in labels)):
            combo = item1+item2
            labels[labels.index(item1)] = combo
            labels.remove(item2)
        return labels

    def splitItem1(self,labels):
        labels.insert(0,labels[0][0])
        labels[1] = labels[1][1:]
        return labels

    def cleanUpLabels(self,labels):
        labels = self.removeItem('c',labels)
        labels = self.removeItem('f',labels) #do before gender, singer separation!
        labels = self.replaceItem('u(1)','u',labels)
        labels = self.replaceItem('a(1)','a',labels)
        labels = self.replaceItem('arepggios','arpeggios',labels)
        labels = self.combine('fast','forte',labels)
        labels = self.combine('fast','piano',labels)
        labels = self.combine('slow','piano',labels)
        labels = self.combine('slow','forte',labels)
        labels = self.combine('lip','trill',labels)
        labels = self.combine('vocal','fry',labels)
        labels = self.splitItem1(labels)

        return labels

    def parseLabels(self,filename):
        info,ext = os.path.splitext(filename)
        if ext=='.csv':
            info = os.path.splitext(info)[0] #remove crepe .f0 tag
        lbl = info.split("_") #known delimiter
        return self.cleanUpLabels(lbl)

    def Create(self,F):
        Singer = [None] * len(F)
        Gender = [None] * len(F)
        VT = [None] * len(F)
        Movement = [None] * len(F)
        Vowel = [None] * len(F)

        for i,f in enumerate(F):
            lbls = self.parseLabels(f)
            #print(lbls)
            Gender[i] = lbls[0]
            Singer[i] = lbls[1]
            Movement[i] = lbls[2]
            VT[i] = lbls[3]
            try:
                Vowel[i] = lbls[4]
            except:
                Vowel[i] = 'N/A'
        print(i+1, "labels generated")
        return Gender, Singer, Movement, VT, Vowel


#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create Corresponding Labels from DAMP Filename Metadata
class DAMPMetadataLabels:
    def __init__(self):
        pass
    def Create(self,F,filepath):
        Followers = [None] * len(F)
        Followees = [None] * len(F)
        Country = [None] * len(F)
        Song = [None] * len(F)

        metadata_all, hlist = self.loadMetadata(filepath)

        for i,f in enumerate(F):
            name = os.path.splitext(f)[0]
            this_metadata = [m for m in metadata_all if (m.split("#"))[0]==name]
            meta_dict = self.parseMetadata(this_metadata[0],hlist)
            Followers[i] = self.logRoundUp(meta_dict['Followers'])
            Followees[i] = self.logRoundUp(meta_dict['Followees'])
            Country[i] = meta_dict['Country']
            Song[i] = meta_dict['Song']
        print(i+1, "labels generated")
        return Followers, Followees, Country, Song

    def loadMetadata(self,filepath):
        with open(filepath) as f:
            metadata_all = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
        metadata_all = [x.strip() for x in metadata_all]
        header = metadata_all[0]
        hlist = header.split("#")
        metadata_all = metadata_all[1:]
        return metadata_all, hlist

    # metadata parsing function for a single point
    def parseMetadata(self,metadata,hlist):
        mlist = metadata.split("#") #known delimiter
        meta_dict = dict(zip(hlist, mlist))
        return meta_dict

    def logRoundUp(self,x):
        if(x!='NULL'):
            x = int(x)
            try:
                y = 10**round(math.log10(int(x)))
                if y > 10000:
                    y = 10000
            except:
                if(x==0): y = 1
                else: y = -1
            return y
        else: return -1



#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create Corresponding Labels from VocalSet Filename Metadata

def CreateArtistLabels(F):
    Artist = [None] * len(F)

    for i,f in enumerate(F):
        name = os.path.splitext(f)[0]
        Artist[i] = f[:2]
    print(i+1, "labels generated")
    return Artist


#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create Corresponding Labels from RAVDESS filename Metadata
class RAVDESSLabels:
    def __init__(self):
        pass

    def getEmotion(self,L):
        if L==1: return 'neutral'
        elif L==2: return 'calm'
        elif L==3: return 'happy'
        elif L==4: return 'sad'
        elif L==5: return 'angry'
        elif L==6: return 'fearful'
        else: return ''

    def parseLabels(self,filename):
        info,ext = os.path.splitext(filename)
        if ext=='.csv':
            info = os.path.splitext(info)[0] #remove crepe .f0 tag
        lbl = info.split("-") #known delimiter
        return lbl

    def Create(self,F):
        Emotion = [None] * len(F)
        Intensity = [None] * len(F)
        Statement = [None] * len(F)
        Actor = [None] * len(F)
        Gender = [None] * len(F)

        for i,f in enumerate(F):
            lbls = self.parseLabels(f)
            #print(lbls)
            Emotion[i] = self.getEmotion(int(lbls[2]))
            Intensity[i] = int(lbls[3])
            Statement[i] = int(lbls[4])
            Gender[i] = int(lbls[6])%2
            Actor[i] = int(lbls[6])
        print(i+1, "labels generated")
        return Emotion, Intensity, Statement, Actor, Gender
