# -*- coding: utf-8 -*-
"""
Helper functions for loading and creating datasets
"""
import numpy as np
import glob
import simplejson
import os
import cv2
import csv
import sys
import unidecode
from PIL import Image
from rename import rename
from helpers import implt
from normalization import letter_normalization
from viz import print_progress_bar

"""
CHARS1 = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ఌ','ఎ', 'ఏ', 'ఐ','ఒ', 'ఓ']
CHARS2 = [ఔ', 'క', 'ఖ', 'గ', 'ఘ', 'ఙ', 'చ', 'ఛ', 'జ', 'ఝ', 'ఞ', 'ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న','ప', 'ఫ', 'బ', 'భ', 'మ', 'య', 'ర', 'ఱ', 'ల', 'ళ', 'ఴ', 'వ', 'శ', 'ష', 'స', 'హ','ౘ', 'ౙ','ౠ', 'ౡ','ఀ', 'ఁ', 'ం', 'ః','ఽ', 'ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ','ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', '్', 'ౕ', 'ౖ', 'ౢ', 'ౣ']
CHAR_SIZE = len(CHARS1)+len(CHARS2)
idxs1 = [i for i in range(10,len(CHARS1)+10)]
idxs2 = [i for i in range(10,len(CHARS2)+10)]

idx_2_chars_1 = dict(zip(idxs1, CHARS1))
idx_2_chars_2 = dict(zip(idxs2, CHARS2))
chars_2_idx_1 = dict(zip(CHARS1, idxs1))
chars_2_idx_2 = dict(zip(CHARS2, idxs2))

def char2idx(s):
    '''l=[]
    for i in range(0,len(s)):
        if s[i] in CHARS1 and s[i+1] in CHARS1:
            l.append(chars_2_idx_1[s[i]])
        elif(s[i] in CHARS1 and s[i+1] in CHARS2):
            j=i+1
            s=''
            while(s[j] not in CHARS2):
                s=s+str(chars_2_idx_2[j]'''
    n=0
    for i in s:
        if i in CHARS1:
            n=chars_2_idx_1[i]
        else:
            n=int(str(n)+str(chars_2_idx_2[i]))
                
            
            
    return n

def idx2char(idx):
    s=''
    for i in range(0,len(str(idx)),2):
        if(i==0):
            s=s+idx_2_chars_1[int(str(idx)[i::2])]
        else:
            s=s+idx_2_chars_2[int(str(idx)[i::2])]
    return s
            
"""
#CHARS=['అంటే','అని','ఈయన','ఎంతో']
CHARS= ['అ','ఆ', 'ఇ','ఈ', 'ఉ','ఊ', 'ఋ','ౠ','ఎ','ఏ', 'ఐ','ఒ','ఓ','కా','క','కి','కీ','కు','కూ','కృ','కౄ','కె','కే','కై','కొ','కో','టే','కౌ','క్','ఔ','క్క','క్కు','క్ట','క్టా','క్టో','ల','డే','ఀ']
CHAR_SIZE = len(CHARS)
idxs = [i for i in range(len(CHARS))]
idx_2_chars = dict(zip(idxs, CHARS))
chars_2_idx = dict(zip(CHARS, idxs))
def char2idx(s):
    try:
        return chars_2_idx[s]
    except:
        print("'"+s+"'",end=',')
        
        
        
def idx2char(n):
    return idx_2_chars[n]

    

def load_words_data(dataloc='data/sets/', is_csv=True, load_gaplines=True):
    """
    Load word images with corresponding labels and gaplines (if load_gaplines == True).
    Args:
        dataloc: image folder location/CSV file - can be list of multiple locations
        is_csv: using CSV files
        load_gaplines: wheter or not load gaplines positions files
    Returns:
        (images, labels (, gaplines))
    """
    print("Loading words...")
    if type(dataloc) is not list:
        dataloc = [dataloc]

    if is_csv:
        #csv.field_size_limit(sys.maxsize)
        length = 0
        for loc in dataloc:
            with open(loc) as csvfile:
                reader = csv.reader(csvfile)
                length += max(sum(1 for row in csvfile)-1, 0)

        labels = np.empty(length, dtype=object)
        images = np.empty(length, dtype=object)
        i = 0
        for loc in dataloc:
            print(loc)
            with open(loc) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    shape = np.fromstring(
                        row['shape'],
                        sep=',',
                        dtype=int)
                    img = np.fromstring(
                        row['image'],
                        sep=', ',
                        dtype=np.uint8).reshape(shape)
                    labels[i] = row['label']
                    images[i] = img
                    
                    #print_progress_bar(i, length)
                    i += 1
    else:
        img_list = []
        tmp_labels = []
        for loc in dataloc:
            tmp_list = glob.glob(os.path.join(loc, '*.png'))
            img_list += tmp_list
            tmp_labels += [name[len(loc):].split("_")[0] for name in tmp_list]

        labels = np.array(tmp_labels)
        images = np.empty(len(img_list), dtype=object)

        # Load grayscaled images
        for i, img in enumerate(img_list):
            #images[i] = cv2.imread(img,0)
            #print(images[i])
            pil_image = Image.open(img).convert('L')
            images[i] = np.array(pil_image)
            # Convert RGB to BGR 
            #os.rename(img[:len(loc)]+"1.png",img[:len(loc)]+temp)
            
            #images[i]=np.array(Image.open(img))

        # Load gaplines (lines separating letters) from txt files
        if load_gaplines:
            gaplines = np.empty(len(img_list), dtype=object)
            for i, name in enumerate(img_list):
                with open(name[:-3] + 'txt', 'r') as fp:
                    gaplines[i] = np.array(simplejson.load(fp))
                
    if load_gaplines:
        assert len(labels) == len(images) == len(gaplines)
    else:
        assert len(labels) == len(images)
    print("-> Number of words:", len(labels))
    
    if load_gaplines:
        return (images, labels, gaplines)
    return (CHAR_SIZE,images, labels)


def _words2chars(images, labels, gaplines):
    """Transform word images with gaplines into individual chars."""
    # Total number of chars
    length = sum([len(l) for l in labels])
    imgs = np.empty(length, dtype=object)
    new_labels = []
    
    height = images[0].shape[0]
    idx = 0;
    for i, gaps in enumerate(gaplines):
        for pos in range(len(gaps) - 1):
            imgs[idx] = images[i][0:height, gaps[pos]:gaps[pos+1]]
            #new_labels.append(char2idx(labels[i][pos]))
            idx += 1
        newlabels.append(char2idx(labels[i]))
           
    print("Loaded chars from words:", length)            
    return imgs, new_labels

def load_chars_data(charloc,RNN=False):
    img_labels=[]
    img_list = glob.glob(os.path.join(charloc, '*.jpeg'))
    img_labels += [name[len(charloc)+1:].split("_")[0] for name in img_list]
    if(RNN==False):

        images = np.zeros((1, 4096))
        labels = []
            # Load grayscaled images
        '''for i, img in enumerate(img_list):
            pil_image = Image.open(img)
            images[i] = np.array(pil_image)'''
        imgs = np.array([letter_normalization(np.array(Image.open(img).convert('L'))) for img in img_list])
        images = np.concatenate([images, imgs.reshape(len(imgs), 4096)])
        for i in img_labels:
            labels.append(char2idx(i))
        labels=np.array(labels)
        images = images[1:]
        print("-> Number of chars:", len(labels))
        return (CHAR_SIZE,images,labels)
    else:
        labels = []
        images = np.empty(len(img_labels), dtype=object)

        for i, img in enumerate(img_list):
            pil_image = Image.open(img)
            images[i] = np.array(pil_image)
        for i in img_labels:
            labels.append(char2idx(i))
        return (CHAR_SIZE,images,labels)
    
        
        
        
    
        

    
    



def load_gap_data(loc='data/processed/breta/word_gaplines', slider=(60, 120), seq=False, flatten=True):
    """ 
    Load gap data from location with corresponding labels.
    Args:
        loc: location of folder with words separated into gap data
             images have to by named as label_timestamp.jpg, label is 0 or 1
        slider: dimensions of of output images
        seq: Store images from one word as a sequence
        flatten: Flatten the output images
    Returns:
        (images, labels)
    """
    print('Loading gap data...')
    #dir_list = glob.glob(os.path.join(loc, "*/"))
    dir_list=glob.glob("G:/Untitled Folder/handwriting-ocr-master/data/processed/breta/words_gaplines")
    dir_list.sort()
    
    if slider[1] > 120:
        # TODO Implement for higher dimmensions
        slider[1] = 120
        
    cut_s = None if (120 - slider[1]) // 2 <= 0 else  (120 - slider[1]) // 2
    cut_e = None if (120 - slider[1]) // 2 <= 0 else -(120 - slider[1]) // 2
    
    if seq:
        images = np.empty(len(dir_list), dtype=object)
        labels = np.empty(len(dir_list), dtype=object)
        
        for i, loc in enumerate(dir_list):
            # TODO Check for empty directories
            img_list = glob.glob(os.path.join(loc, '*.png'))
            if (len(img_list) != 0):
                img_list = sorted(imglist, key=lambda x: int(x[len(loc):].split("_")[1][:-4]))
                images[i] = np.array([(cv2.imread(img, 0)[:, cut_s:cut_e].flatten() if flatten else
                                       cv2.imread(img, 0)[:, cut_s:cut_e])
                                      for img in img_list])
                labels[i] = np.array([int(name[len(loc):].split("_")[0]) for name in img_list])
        
    else:
        images = np.zeros((1, slider[0]*slider[1]))
        labels = []

        for i in range(len(dir_list)):
            img_list = glob.glob(os.path.join(dir_list[i], '*.png'))
            if (len(img_list) != 0):
                imgs = np.array([cv2.imread(img, 0)[:, cut_s:cut_e] for img in img_list])
                images = np.concatenate([images, imgs.reshape(len(imgs), slider[0]*slider[1])])
                labels.extend([int(img[len(dirlist[i])]) for img in img_list])

        images = images[1:]
        labels = np.array(labels)
		
    if seq:
        print("-> Number of words / gaps and letters:",
              len(labels), '/', sum([len(l) for l in labels]))
    else:
        print("-> Number of gaps and letters:", len(labels))
    return (images, labels)    


def corresponding_shuffle(a):
    """ 
    Shuffle array of numpy arrays such that
    each pair a[x][i] and a[y][i] remains the same.
    Args:
        a: array of same length numpy arrays
    Returns:
        Array a with shuffled numpy arrays
    """
    assert all([len(a[0]) == len(a[i]) for i in range(len(a))])
    p = np.random.permutation(len(a[0]))
    for i in range(len(a)):
        a[i] = a[i][p]
    return a


def sequences_to_sparse(sequences):
    """
    Create a sparse representention of sequences.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
        
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape
