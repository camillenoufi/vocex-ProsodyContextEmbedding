from PseudoTaskModel import *
from SlotFillModel import *
from embed_utils import *
from keras import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import numpy as np
import time

# Initialize parser
msg = 'Adding Instructions'
parser = argparse.ArgumentParser(description = msg)
# Adding optional argument
parser.add_argument("-d", "--DataDir", help = "Data and File List Directory")
parser.add_argument("-f", "--Filename", help = "Name of contour file list")
parser.add_argument("-m", "--ModelName", help = "Name of Trained Model (for .ckpt loading)")
parser.add_argument("-m2", "--Model2Name", help = "Name of 2nd Trained Model (for .ckpt loading), for fusion")
parser.add_argument("-m3", "--Model3Name", help = "Name of 2nd Trained Model (for .ckpt loading), for fusion")
parser.add_argument("-n", "--NumEmbeddings", help = "Number of Embeddings to compute")
parser.add_argument("-s", "--SaveFilename", help = "File to save embeddings")
parser.add_argument("-bs", "--BatchSize", help = "Number of continguous contours to embed and create temporal fusion, default = 1")
parser.add_argument("-hs", "--HopSize", help = "Continguous contours overlap (non-zero decimal) to embed and create temporal fusion, default = 1")
parser.add_argument("-tf", "--ThresholdFlag", help = "if using a model trained on thresholded contours, default 0")


def InitializeArray(n):
    return [None] * n

def GetEmbedderInstance(model):
    if model[-2:] == 'sf':
        return SFContourEmbedder(checkpoint = '{0}.ckpt'.format(model))
    else:
        return PTContourEmbedder(checkpoint = '{0}.ckpt'.format(model))

def main(datadir, filename, n, batch_size, hop_size, savefile, threshold_flag, model, model2,model3):
    embedder = []
    g1 = tf.Graph()
    with g1.as_default():
        emb1 = GetEmbedderInstance(model)
        embedder.append(emb1)
    if model2:
        g2 = tf.Graph()
        with g2.as_default():
            emb2 = GetEmbedderInstance(model2)
            embedder.append(emb2)
    if model3:
        g3 = tf.Graph()
        with g3.as_default():
            emb3 = GetEmbedderInstance(model3)
            embedder.append(emb3)

    tic = time.time()
    print("Beginning Computation...")

    if(threshold_flag==0):
        print("TF = {0}, Computing Random contour embeddings....".format(threshold_flag))
        #rce = RandomContourExtractor(datadir = datadir,filelist=filename)
        npz = np.load('./embeddings/{0}_bs{1}_contours.npz'.format(filename,batch_size))
        ContourBatchList = npz['CBL']
        StartFrameList = npz['SFL']
        FileList = npz['FL']
        Embeddings = InitializeArray(n)
        Contours = InitializeArray(n)
        StartFrames = InitializeArray(n)
        OriginalFileName = InitializeArray(n)
        nan_contours = 0
        for i in range(n):
            #contour_batch, sf_batch, f = rce.GetRandomContour(batch_size,hop_size) #returns contour in cents, start frame, and csv file
            nan_flag, e_batch, c_batch = ComputeEmbedding(ContourBatchList[i],embedder,threshold_flag)
            e_batch = np.array(e_batch)
            c_batch = np.array(c_batch)
            if nan_flag == 0:
                OriginalFileName[i] = FileList[i]
                Embeddings[i] = e_batch
                Contours[i] = c_batch
                StartFrames[i] = StartFrameList[i]
            else:
                nan_contours += 1
            if(i%100 == 0):
                print("S",i)

    else:
        print("TF = {0}, Computing thresholded contour embeddings".format(threshold_flag))
        tce = ThresholdedContourExtractor(datadir = datadir,filelist=filename)
        Embeddings = []
        Contours = []
        StartFrames = []
        OriginalFileName = []
        nan_contours = 0
        for i in range(n):
            contour_batch, sf_batch, f = tce.GetContourList(batch_size,hop_size)
            n_contours = len(contour_batch)
            for j in range(n_contours):
                nan_flag, e_batch, c_batch = ComputeEmbedding(contour_batch[j],embedder,threshold_flag)
                e_batch = np.array(e_batch)
                c_batch = np.array(c_batch)
                if nan_flag == 0:
                    OriginalFileName.append(f)
                    Embeddings.append(e_batch)
                    Contours.append(c_batch)
                    if batch_size == 1:
                        StartFrames.extend(sf_batch)
                    else:
                        StartFrames.extend(sf_batch[j])
                else:
                    nan_contours += 1
            if(i%50 == 0):
                print("Batch",i)

    print("Num created:",len(Embeddings))
    print("Num embeddings skipped:", nan_contours)
    print(n,"Saving...")
    np.savez_compressed(savefile, C=Contours, SF=StartFrames, F=OriginalFileName, E=Embeddings)

    toc = time.time() - tic
    print("Sucess! Embeddings saved to: {0}.npz".format(savefile))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.DataDir:
        datadir = args.DataDir
    else:
        datadir = defaultDataDir = '/Users/camillenoufi/cnoufi/Research/VocEx/winter2020/data_by_singer/all/csv/test/'

    if args.Filename:
        filename = args.Filename
    else:
        filename = 'VocalSet_testList.txt'

    if args.ModelName:
        model = args.ModelName
    else:
        model = 'vs-r-a0-perf'

    if args.Model2Name:
        model2 = args.Model2Name
    else:
        model2 = []

    if args.Model3Name:
        model3 = args.Model3Name
    else:
        model3 = []

    if args.ThresholdFlag:
        thf = int(args.ThresholdFlag)
    else:
        thf = 0

    if args.NumEmbeddings:
        n = int(args.NumEmbeddings)
    else:
        n = 10000

    if args.BatchSize:
        bs = int(args.BatchSize)
    else:
        bs = 1

    if args.HopSize:
        hs = float(args.HopSize)
        if hs<=0 or hs>2:
            hs = 1
            print("hop size adjusted to 1.  bounds (0-2]")
    else:
        hs = 1

    if args.SaveFilename:
        savefile = args.SaveFilename
    else:
        if model2 and model3:
            savefile = './embeddings/{0}+{1}+{2}_bs{3}_{4}_embs'.format(model,model2,model3,bs,filename)
        elif model2 and not model3:
            savefile = './embeddings/{0}+{1}_bs{2}_{3}_embs'.format(model,model2,bs,filename)
        else:
            savefile = './embeddings/{0}_bs{1}_{2}_embs'.format(model,bs,filename)

    main(datadir, filename, n, bs, hs, savefile, thf, model, model2, model3)
