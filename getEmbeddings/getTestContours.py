from embed_utils import *
import argparse
import numpy as np
import time

# Initialize parser
msg = 'Adding Instructions'
parser = argparse.ArgumentParser(description = msg)
# Adding optional argument
parser.add_argument("-d", "--DataDir", help = "Data and File List Directory")
parser.add_argument("-f", "--Filename", help = "Name of contour file list")
parser.add_argument("-n", "--NumEmbeddings", help = "Number of Embeddings to compute")
parser.add_argument("-s", "--SaveFilename", help = "File to save contours")
parser.add_argument("-bs", "--BatchSize", help = "Number of continguous contours to embed and create temporal fusion, default = 1")
parser.add_argument("-hs", "--HopSize", help = "Continguous contours overlap (non-zero decimal) to embed and create temporal fusion, default = 1")

def main(datadir, filename, n, batch_size, hop_size, savefile):
    rce = RandomContourExtractor(datadir = datadir,filelist=filename)

    ContourBatchList = [None] * n
    FileList = [None] * n
    StartFrameList = [None] * n

    tic = time.time()
    for i in range(n):
        ContourBatchList[i], StartFrameList[i], FileList[i] = rce.GetRandomContour(batch_size,hop_size) #returns contour in cents, start frame, and csv file
        if i%100==0:
            print(i)

    np.savez_compressed(savefile, CBL=ContourBatchList, FL=FileList, SFL=StartFrameList)
    print("contours saved to",savefile)
    print("elapsed:",time.time() - tic)

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

    if args.NumEmbeddings:
        n = int(args.NumEmbeddings)
    else:
        n = 100

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
        savefile = './embeddings/{0}_bs{1}_contours'.format(filename,bs)

    main(datadir, filename, n, bs, hs, savefile)
