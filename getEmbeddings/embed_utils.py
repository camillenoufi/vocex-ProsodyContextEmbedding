import os
import numpy as np
np.random.seed(0)

def ComputeEmbedding(contour_batch,embedder,threshold_flag):
    nan_flag = 0
    e_batch = []
    c_batch = []
    batch_size = contour_batch.reshape(len(contour_batch),1).shape[1]
    for j in range(batch_size):
        if(batch_size==1):
            if(threshold_flag):
                this_contour = contour_batch
            else:
                this_contour = contour_batch[0]
        else:
            this_contour = contour_batch[j]
        #if( 1 not in ([1 if (np.isnan(c) or np.isinf(c)) else 0 for c in this_contour]) ):
        e = np.empty(0)
        for M in embedder:
            e = np.concatenate((e, M.embedContour(this_contour)))
        e_batch.extend(e)
        c_batch.extend(this_contour)
        #else:
            #nan_flag = 1
            #break
    return nan_flag, e_batch, c_batch


class RandomContourExtractor:
    def __init__(self, datadir = '', filelist = ''):
        self.datadir = datadir
        self.filelist = filelist
        self.content = self.GetCSVList()

    def GetCSVList(self):
        with open(os.path.join(self.datadir,self.filelist)) as f:
            content=f.readlines()
        print(len(content))
        print(content[0][:-1])
        return content

    def GetRandomContour(self,batch_size,hop_size):
        # Choose a random csv
        t1=np.random.randint(0,len(self.content))
        currentfile=self.content[t1][:-1]

        # loading the f0 track
        currentpath = os.path.join(self.datadir, currentfile)
        ext = os.path.splitext(currentpath)[1]

        if(ext == '.csv'):
            currentpitch=self.csv2pitch(currentpath)
        elif(ext == '.npy'):
            currentpitch=self.npy2pitch(currentpath)
        elif(ext == '.npz'):
            currentpitch=self.npz2pitch(currentpath, "pitch")
        else:
            print('file format not supported')
            pass

        # convert it to cents
        c = self.freq2cents(currentpitch)

        # get all contours
        chunk_batch, start_batch = self.getContourBatch(c,batch_size,hop_size)

        return chunk_batch, start_batch, currentfile

    def getContourBatch(self,c,batch_size,hop_size):
        CLEN = 100
        chunk_batch = []
        start_batch = []
        try:
            start=np.random.randint(0,len(c)-int(batch_size*CLEN*hop_size)-1)
        except:
            start = 0
        for i in range(batch_size):
            # Choose a starting point: You can do it anywhere
            start_batch.append(start)
            end = start+CLEN
            try:
                chunk=self.postProcess(np.array(c[start:end])) # GEt the chunk
                assert chunk.shape[0]==100
            except: #in case the file is less than 100 frames
                subchunk = c[start:]
                chunk=np.concatenate((self.postProcess(np.array(subchunk)), np.zeros(CLEN-len(subchunk)))) # GEt the chunk
                assert chunk.shape[0]==100
            chunk_batch.append(chunk)
            ## MAKE SURE THAT NO NAN OR INF PASS THROUGH IN ANY OF THE PITCH CONTOURS
            start += int(hop_size*CLEN)
        return chunk_batch, start_batch


    def csv2pitch(self,fname):
        with open(fname) as cf:
            pitchlist=cf.readlines()
        currentpitch=np.zeros((len(pitchlist),))
        for i in range(1,len(pitchlist)):
            if np.float(pitchlist[i].split(',')[2])>0.3:
                currentpitch[i-1]=pitchlist[i].split(',')[1]
        return currentpitch

    def npy2pitch(self,fname):
        return np.load(fname)

    def npz2pitch(self,fname,key):
        pt = np.load(fname)[key]
        try:
            currentpitch = pt[500:(len(pt)-500)] #if file is less than ~11 seconds long
        except:
            currentpitch = pt[100:(len(pt)-100)] #try chopping ~1sec off each end instead
        return currentpitch


    def freq2cents(self,freqs):
        cents = [ 1200*np.log2(f/55) for f in freqs]
        return cents

    def postProcess(self,c):
        for j in range(0,len(c)):
            if np.isnan(c[j])==1:
                c[j]=np.median(c)
            if np.isinf(c[j])==1:
                c[j]=np.median(c)
        c=c-np.median(c)
        return c


class ThresholdedContourExtractor:
    def __init__(self, datadir = '', filelist = ''):
        self.datadir = datadir
        self.filelist = filelist
        self.content = self.GetCSVList()

    def GetCSVList(self):
        with open(os.path.join(self.datadir,self.filelist)) as f:
            content=f.readlines()
        print(len(content))
        print(content[0][:-1])
        return content

    def GetContourList(self,batch_size,hop_size):
        # Choose a random csv
        t1=np.random.randint(0,len(self.content))
        currentfile=self.content[t1][:-1]

        # loading the f0 track
        currentpath = os.path.join(self.datadir, currentfile)
        ext = os.path.splitext(currentpath)[1]

        if(ext == '.csv'):
            currentpitch=self.csv2pitch(currentpath)
        elif(ext == '.npy'):
            currentpitch=self.npy2pitch(currentpath)
        elif(ext == '.npz'):
            currentpitch=self.npz2pitch(currentpath, "pitch")
        else:
            print('file format not supported')
            pass

        # convert it to cents
        c = self.freq2cents(currentpitch)

        # get all contours
        slices, start_frames = self.getContourSlices(c)
        chunk_batch = [None] * (len(slices)-batch_size)
        start_batch = [None] * (len(slices)-batch_size)
        if(batch_size > 1):
            chunk_batch = [slices[i:(i+batch_size)] for i,x in enumerate(slices) if i<(len(slices)-batch_size)]
            start_batch = [start_frames[i:(i+batch_size)] for i,x in enumerate(start_frames) if i<(len(start_frames)-batch_size)]
        else:
            chunk_batch = slices
            start_batch = start_frames
        return chunk_batch, start_batch, currentfile

    def calcDerivative(self,c):
        UPPER = 45
        LOWER = -1*UPPER
        NAN_FILL = LOWER - 1

        dc = np.diff(c)

        #corrections
        idxPos = [i for i,x in enumerate(dc) if x >= UPPER]
        idxNeg = [i for i,x in enumerate(dc) if x <= LOWER]
        idxNan = [i for i,x in enumerate(dc) if np.isnan(x)]
        dc[idxPos] = UPPER
        dc[idxNeg] = LOWER
        dc[idxNan] = NAN_FILL

        return dc,UPPER,LOWER,NAN_FILL

    def getContourSlices(self,c):
        K_min = 15
        K_MAX = 100

        slices = []
        start_frames = []

        dc,UPPER,LOWER,NAN_FILL = self.calcDerivative(c)
        # segment track into contour sections
        i = 0
        while ( i <  (len(dc) - K_min) ):
            frame_size = 0
            # while derivative is "in bounds" and contour is less than K_max frames
            while ( (i+frame_size) < (len(dc)-1) and dc[i+frame_size] < UPPER and dc[i+frame_size] > LOWER):
                frame_size += 1
            #only append contours greater than K_min frames
            if (frame_size >= K_min):
                if(frame_size > K_MAX): #added to truncate > 100
                  frame_size = K_MAX
                # append original pitch contour (variable length <= 350) to master cents slice list
                chunk = c[i:i+frame_size]
                chunk = chunk - np.median(chunk)
                chunk = np.pad(chunk, (0, K_MAX-chunk.shape[0]), 'constant', constant_values=0)
                slices.append( chunk ) #all contour slices will start at 0

                #append location and length info
                start_frames.append(i)
            # move to beginning of next frame start
            i = i + frame_size + 1

        return slices, start_frames

    def csv2pitch(self,fname):
        with open(fname) as cf:
            pitchlist=cf.readlines()
        currentpitch=np.zeros((len(pitchlist),))
        for i in range(1,len(pitchlist)):
            if np.float(pitchlist[i].split(',')[2])>0.3:
                currentpitch[i-1]=pitchlist[i].split(',')[1]
        return currentpitch

    def npy2pitch(self,fname):
        return np.load(fname)

    def npz2pitch(self,fname,key):
        pt = np.load(fname)[key]
        try:
            currentpitch = pt[500:(len(pt)-500)] #if file is less than ~11 seconds long
        except:
            currentpitch = pt[100:(len(pt)-100)] #try chopping ~1sec off each end instead
        return currentpitch


    def freq2cents(self,freqs):
        cents = [ 1200*np.log2(f/55) for f in freqs]
        return cents
