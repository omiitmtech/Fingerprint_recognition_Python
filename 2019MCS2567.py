#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin
import numpy as np
import cv2 
from scipy import ndimage
from scipy import signal
import warnings
import math
import scipy.ndimage
import scipy
import numpy as np
from PIL import Image 
import math
import cv2 as cv


# In[2]:


def normalise(img):
    mean = np.mean(img)
    normed = img - mean
    std = np.std(img)
    normed/=std    
    return(normed)


# In[3]:


def std_im(im,v_stddevim, start_index,rows,cols,thresh):
    stddevim = v_stddevim[start_index:rows][:,start_index:cols]
    rm_v     = 0.5
    mask     = stddevim > thresh
    n_mask   =   im[mask]  
    mean_val = np.mean(n_mask)
    normim   = (im - mean_val)/(np.std(n_mask))
    return(normim,mask)


# In[4]:


def read_images2(v_images_path):
    images_matrix = []
    image_files = os.listdir(v_images_path)
    for image in tqdm(sorted(image_files)):
        print(image,end =" ")
        images_matrix.append(cv.imread(v_images_path+image,0))
    return np.array(images_matrix,dtype='float64')     


# In[5]:


def divide_in_k_blocks(v_images_matrix,n_rows, n_cols,w_row,w_col):
    k_img_matrices = v_images_matrix.reshape(n_rows//w_row, w_row, -1, w_col).swapaxes(1,2).reshape(-1, w_row, w_col)
    return k_img_matrices


# In[6]:


#to calculate grey-level variance of kth level
def gray_level_variance(v_kth_block,v_w):
    return np.sum(np.square(v_kth_block - np.mean(v_kth_block)))/(v_w**2)    


# In[7]:


def segmentation(v_images_matrix,v_w,v_threshold):
    v_no_rows, v_no_cols = v_images_matrix.shape
    v_images_matrix = normalise(v_images_matrix)
    segmented_image = v_images_matrix.copy()
    segmented_image = np.ones(v_no_rows*v_no_cols)
    segmented_image.resize(v_no_rows,v_no_cols)
    image_threshold = v_threshold*(np.std(v_images_matrix))
    i=0
    while i<v_no_rows:
        j=0
        while j<v_no_cols:
            sub_matrix = v_images_matrix[i:i+v_w,j:j+v_w]
            gray_level_var = gray_level_variance(sub_matrix,v_w)
#             print(gray_level_var,image_threshold)
            if  gray_level_var < image_threshold: 
                segmented_image[i:i+v_w,j:j+v_w]=0
            j+=v_w
        i+=v_w
    segmented_image*=v_images_matrix
    return segmented_image*100


# In[8]:


def calculate_mean(v_images_matrix,N):
    return (np.sum(v_images_matrix)/(N**2))

def calculate_variance(v_images_matrix,N):
    M = calculate_mean(v_images_matrix,N)
    return np.sum(np.square(v_images_matrix - M))/(N**2) 


# In[9]:


def normalization(I,M0,V0):
    N = I.shape[0]
    one_matrix = np.ones(N*N)
    one_matrix.resize(N,N)
    Normalized_image = one_matrix.astype(dtype='uint8')
    M   = calculate_mean(I,N)
    V   = calculate_variance(I,N)
    i=0
    j=0
    while i<N:
        while j<N:
            temp_val = np.sqrt((V0*((I[i][j] - M)**2))/V)
            if(I[i][j] > M):
                Normalized_image[i][j] = M0+temp_val
            else:
                Normalized_image[i][j] = M0-temp_val
            j+=1
        i+=1
    print('Image has been normalized successfully!')
    return Normalized_image


# In[30]:


def norm(img_file):
    img = Image.open(img_file)
    (rows, cols) = img.size
    img2 = cv2.imread(img_file, 0)
    desired_mean = 190.0
    desired_var = 150.0
    rows,cols = img2.shape
    image_mean = np.mean(img2)
    image_var = np.std(img2) ** 2
    normalised_img = img2.copy()
    i = 0
    while i < cols:
        j = 0
        while j < rows:
            if img2[i, j] > image_mean:
                normalised_img[i, j] = desired_mean + np.sqrt(desired_var *((img2[i, j]-image_mean)**2) / image_var)
            else:
                normalised_img[i, j] = desired_mean - np.sqrt(desired_var *((img2[i, j]-image_mean)**2) / image_var)
            j += 1
        i += 1
    

    #cv.imshow('normalised_img', normalised_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return normalised_img


# In[11]:


def show_image(v_image):
    image = v_image.astype(dtype='uint8')
    cv.imshow('segmented image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# In[12]:


def cal_ridge_segment(im,blksze,thresh,v_rt):
    rows = im.shape[0]
    cols = im.shape[1] 
    denom1 = blksze*1.0
    neum1 = rows*1.0
    mult1 = np.ceil(neum1/denom1)
    new_rows =  np.int(blksze * mult1)

    denom2 = blksze*1.0
    neum2  = cols*1.0
    mult2  = np.ceil(neum2/denom2)
    new_cols =  np.int(blksze * mult2)

    im = normalise(im)
    padded_img = np.zeros(new_rows*new_cols)
    padded_img.resize(new_rows,new_cols)

    stddevim = np.zeros(new_rows*new_cols)
    start_index = 0
    stddevim.resize(new_rows,new_cols)
    
    padded_img[start_index:rows][:,start_index:cols] = im
    i=0
    while i < new_rows:
        j=0
        while j < new_cols:
            block = padded_img[i:i+blksze][:,j:j+blksze]
            i_index = i+blksze
            ones_block = np.ones(block.shape)
            temp_val = np.std(block)*ones_block
            j_index = j+blksze
            stddevim[i:i_index][:,j:j_index] = temp_val
            j=j+blksze
        i=i+blksze
    res_normi = std_im(im,stddevim,0,rows,cols,thresh)
    return(res_normi[0],res_normi[1])


# In[13]:


#ridge_orient.py
def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma,val_rt):
    ut=2
    rows = im.shape[0]
    cols = im.shape[1]
    sze = np.trunc(6*gradientsigma)
    if sze%ut == 0:
        sze+=1
    sze = int(sze)
    gauss = cv2.getGaussianKernel(sze,gradientsigma)
    f = gauss * gauss.T
    fy = np.gradient(f)[0]
    fx = np.gradient(f)[1]
    Gx = signal.convolve2d(im,fx,mode='same')    
    Gxx = Gx**ut
    Gxx = ndimage.convolve(Gxx,f)
    Gy = signal.convolve2d(im,fy,mode='same')
    Gyy = Gy**ut
    Gyy = ndimage.convolve(Gyy,f)
    sze = np.trunc(6*blocksigma)
    sze = int(sze)
    gauss = cv2.getGaussianKernel(sze,blocksigma)
    um=6
    f = gauss * gauss.T
    ep = np.finfo(float).eps
    Gxy = Gx*Gy
    Gxy = ut*ndimage.convolve(Gx*Gy,f)
    tpp = np.sqrt((Gxy**ut))
    ppn = ((Gxx - Gyy)**ut)

    denom = tpp + ppn
    um*=ut
    denom += ep 
    sin2theta = Gxy/denom  
    diff_gxxgyy = Gxx-Gyy    
    cos2theta = diff_gxxgyy/denom
    if orientsmoothsigma:
        sze = np.trunc(6*orientsmoothsigma)
        if sze % ut == 0:
            sze = sze+1    
        int_sze = int(sze)
        gauss = cv2.getGaussianKernel(int_sze,orientsmoothsigma)
        int_sze+=um
        f = gauss * gauss.T
        pi_n = int_sze*2
        cos2theta = ndimage.convolve(cos2theta,f)
        po = np.sqrt(pi_n) 
        sin2theta = ndimage.convolve(sin2theta,f) 

        rte = np.pi/ut
        rem = np.arctan2(sin2theta,cos2theta)
        rem/=ut
    return(rte+rem)


# In[14]:


def frequest(im,orientim,windsze,minwlen,mxwlen):
    t2   = 2
    orientval1 = np.cos(t2*orientim)
    orientval2 = np.sin(t2*orientim)
    rows = im.shape[0]
    orient = math.atan2(np.mean(orientval2),np.mean(orientval1))
    orient/=2
    cols = im.shape[1]
    param_val = orient/np.pi*180 + 90
    t_false = False
    val_st = rows/math.sqrt(2)
    rotim = scipy.ndimage.rotate(im,param_val,axes=(1,0),reshape = t_false,order = 3,mode = 'nearest')   
    km = int(np.trunc(val_st))
    temp_val = rows- km
    temp = 0
    rotim = rotim[int(np.trunc((temp_val)/2)):int(np.trunc((temp_val)/2))+int(np.trunc(val_st))][:,int(np.trunc((temp_val)/2)):int(np.trunc((temp_val)/2))+int(np.trunc(val_st))]
    ones = np.ones(windsze)
    proj = np.sum(rotim,axis = 0)
    peak_thresh = 2   
    dilation = scipy.ndimage.grey_dilation(proj, windsze,structure=ones)

    
    if temp < 0:
        pass
    else:
         temp = abs(proj - dilation)

    bool1 = temp<peak_thresh
    mean_proj = np.mean(proj)
    bool2 = proj > mean_proj
    maxpts =  bool1 & bool2
    rt=0.3
    maxind = np.where(maxpts)
    rt+=temp
    rows_maxind = np.shape(maxind)[0]
    cols_maxind = np.shape(maxind)[1]

    if(cols_maxind<2):
        freqim = np.zeros(rows*cols)
        freqim.resize(rows,cols)
    else:
        NoOfPeaks = temp
        rt -=10
        neuom = maxind[0][cols_maxind-1] - maxind[0][0]
        waveLength = neuom/(cols_maxind - 1)

        if waveLength>=minwlen and waveLength<=mxwlen:
            NoOfPeaks = cols_maxind-1
            ones_shape = np.ones(im.shape)
            freqim = 1/np.double(waveLength) * ones_shape 
        else:
            rt+=temp
            r,c=im.shape 
            freqim = np.zeros((r,c))
        
    return(freqim)
    


# In[15]:


def ridge_freq(im, mask, orient, blksze, windsze,minwlen, mxwlen):
    rows = im.shape[0]
    cols = im.shape[1]
    freq = np.zeros(rows*cols)
    freq.resize(rows,cols)
    r = 0
    c = 0
    row_limit = rows - blksze 
    c_m=0.8
    col_limit = cols-blksze
    while r < row_limit:
        c=0
        while c < col_limit:
            r_l = r+blksze
            c_l = c+blksze
            blkim = im[r:r_l][:,c:c_l]
            blkor = orient[r:r_l][:,c:c_l]
            fre_val = frequest(blkim,blkor,windsze,minwlen,mxwlen)
            freq[r:r_l][:,c:c_l] = fre_val 
            c = c + blksze
        r = r + blksze

    
    freq = freq*mask
    r_m = 0.56
    freq_1d = np.reshape(freq,(1,rows*cols))
    c=r_m*c_m
    ind = np.where(freq_1d>0)
    ind = np.array(ind)
    rows+=r_m
    ind = ind[1,:]        
    cols+=c_m
    meanfreq = np.mean(freq_1d[0][ind])     
    return(freq,np.mean(freq_1d[0][ind]))


# In[16]:


def make_grid(p_val,n_val):
    v = 2*p_val
    v+=1
    s = np.linspace(n_val,p_val,v)
    return np.meshgrid(s,s)

def ridge_filter(im, orient, freq, kx, ky):
    
    im   = np.double(im)
    rows = im.shape[0]
    cols = im.shape[1]
    newim = np.zeros(rows*cols)
    newim.resize(rows,cols)
    
    n_n = rows*cols
    freq_1d = np.reshape(freq,(1,n_n))

    ind = np.array(np.where(freq_1d>0))
    angleInc = 3


    ind = ind[1,:]       
    unfreq = np.unique(np.double(np.round((freq_1d[0][ind] *100)))/100)
    rt_val = np.max([1/unfreq[0]*kx,1/unfreq[0]*ky])
    rt_val *=3
    sze2 = np.round(rt_val)
    sigmax = 1/unfreq[0]*kx
    sze = sze2.astype('int64')

    x,y = make_grid(sze,-sze)
    sigmay = 1/unfreq[0]*ky

    r_rtm = np.cos(2*np.pi*unfreq[0]*x)
    reffilter = np.exp(-(( (x**2)/(sigmax**2) + (np.power(y,2))/(sigmay**2)))) * r_rtm  
    m_valx = 0.3
    filt_rows = reffilter.shape[0]
    filt_cols =  reffilter.shape[1] 
    gabor_filter = np.array(np.zeros((int(180/angleInc),int(filt_rows),int(filt_cols))))

    omn=0
    m_limit = int(180/angleInc)
    while omn < m_limit:
        t_vale = -(omn*angleInc + 90)
        t_false =False
        gabor_filter[omn] = scipy.ndimage.rotate(reffilter,t_vale,reshape = t_false)
        omn = omn +1

    maxsze = int(sze)  
    omn+=2 
    temp = freq>0    


    validr,validc = np.where(temp)    
    final_temp = True
    final_temp = final_temp & (validr>maxsze) & (validr<rows - maxsze) & (validc>maxsze) & (validc<cols - maxsze) & final_temp    
    finalind = np.where(final_temp) 

    one8 = 180
    orientindex = np.round(orient/np.pi*180/angleInc)
    i=0
    j=0
    while i < rows:
        j=0
        while j <cols:
            o_val = orientindex[i][j] 
            if(o_val < 1):
                orientindex[i][j] = o_val + np.round(one8/angleInc)
            if(o_val > np.round(one8/angleInc)):
                orientindex[i][j] = o_val - np.round(one8/angleInc)
            j= j+1
        i=i+1
    finalind_rows,finalind_cols = np.shape(finalind)
    sze = int(sze)
    k=0
    while k < finalind_cols:
        r = validr[finalind[0][k]]
        l1index = r-sze
        r1index = r+sze + 1
        c = validc[finalind[0][k]]  
        l2index = c-sze
        r2index = c+sze + 1
        img_block = im[l1index:r1index][:,l2index:r2index]
        oindex = orientindex[r][c]
        oindex = int(oindex) 
        oindex-=1
        newim[r][c] = np.sum(img_block * gabor_filter[oindex])
        k = k+1
        
    return(newim)    


# In[17]:


#app.py
def removedot(invertThin):
    temp1 = invertThin/255
    i=0
    j=0
    temp2 = temp1.copy()
    filter0 = numpy.zeros(100)
    filter0.resize()
    W,H = invertThin.shape[:2]
    filtersize = 6
    th = 3
    ret=0
    temp_matrix = numpy.zeros(36)
    temp_matrix.resize(6,6)
    
    while i < (W - filtersize):
        while j < (H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]
            flag = 0
            s1 = np.sum(filter0[:,0])
            s2 = np.sum(filter0[0,:]) 
            s3 = np.sum(filter0[:,filtersize - 1])
            s4 = np.sum(filter0[filtersize - 1,:])
            if (s1==0 or s2==0 or s3==0 or s4==0):
                flag +=1
            ret+=1
            if flag > th:
                ret+=1
                temp2[i:i + filtersize, j:j + filtersize] = temp_matrix
            j+=1
        i+=1

    return temp2


# In[18]:


#app.py
def descriptors(img):
    clip_val = float(2)
    clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(8,8))
    threshold_harris = 125
    keypoints = []
    blksze = 16
    thresh = 0.1
    img = clahe.apply(img)
    ridge_segment_res = cal_ridge_segment(img,16,0.1,0.5)
    normim = ridge_segment_res[0]
    mask   = ridge_segment_res[1]
    orientim = ridge_orient(normim, 1, 7, 7,0.3)
    freq,medfreq = ridge_freq(normim, mask, orientim, 38, 5, 5,15)

    freq = medfreq*mask
    newim = ridge_filter(normim, orientim, freq,0.65, 0.65)

    img = newim < -3
    
    img = img.astype(dtype='uint8')
    val1= 127
    val2=  val1*2+1
    ret, img = cv2.threshold(img, val1, val2, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    val_p = 4*0.01
    img[img == val2] = 1
    m_val = 3
    skeleton = skeletonize(img)
    skeleton = skeleton.astype(dtype='uint8')
    # skeleton = numpy.array(skeleton, dtype=numpy.uint8)
    skeleton = removedot(skeleton)
    val_rest = 0.8
    harris_corners = cv2.cornerHarris(img, m_val, m_val, val_p)
    v_z = 0
    vm_c= 100
    h_norm = cv2.normalize(harris_corners, v_z, val2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    x_val = h_norm.shape[0]
    y_val = h_norm.shape[1]
    x=0
    y=0
    for x in range(v_z, x_val):
        for y in range(v_z,y_val):
            vm_c *=2
            t_val = h_norm[x][y] 
            if t_val> threshold_harris:   
                keypoints.append(cv2.KeyPoint(y, x, 1))
    # Define descriptor
    orb = cv2.ORB_create()
    vm_c *=2
    vm_c +=v_z
    pres, des = orb.compute(img, keypoints)
    return (keypoints, des)


# In[19]:


image_database = 'database/'


# In[20]:


def read_image(v_images_path,image_name,image_mode):
    return cv2.imread(v_images_path + image_name, cv2.IMREAD_GRAYSCALE)


# In[21]:


def print_both_images(img1,img2,name1,name2):
    f, axarr = plt.subplots(1,2)
    print('image 1:',name1)
    axarr[0].imshow(img1)
    print('image 2:',name2)
    axarr[1].imshow(img2)
    plt.savefig('output/'+name1[0:-4]+'_and_'+name2[0:-4]+'.png')


# In[22]:


def print_image(img,name1,name2):
    plt.imshow(img)
    cv2.imwrite('output/'+name1[0:-4]+'_and_'+name2[0:-4]+'_match.png',img)


# In[23]:


def write_image(img_name,ltype,v_image):
    cv.imwrite('output/'+img_name[0:-4]+'_'+ltype+'_'+'.png',v_image)


# In[24]:


def main_fun(image_path,train_image,test_image,th):
    img1 = read_image(image_path,train_image,cv2.IMREAD_GRAYSCALE)
    print('image reading...')
    kp1,des1  = descriptors(img1)
    write_image(train_image,'after_normalization',norm(image_path+train_image))
    write_image(test_image,'after_normalization',norm(image_path+test_image))
    img2 = read_image(image_path,test_image,cv2.IMREAD_GRAYSCALE)
    print('image is getting prepared for matching...')
    kp2,des2  =  descriptors(img2)
    write_image(train_image,'after_segmentation',segmentation(img1,30,0.1))
    write_image(test_image,'after_segmentation',segmentation(img2,30,0.1))
    print('Fingerprints Matching...')

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
    print('calculating score....')
    score = 0;
    for match in matches:
        score += match.distance
    score_threshold = th
    matching_score = score/len(matches)
    if matching_score < score_threshold:
        print('Matching Score :',matching_score)
        print("Fingerprint matches.")
    else:
        print('Matching Score :',matching_score)
        print("Fingerprint does not match.")

    train_image_keypoints = cv2.drawKeypoints(img1, kp1, outImage=None)
    test_image__keypoints = cv2.drawKeypoints(img2, kp2, outImage=None)
    print_both_images(train_image_keypoints, test_image__keypoints,train_image,test_image)
    matches_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
    print_image(matches_image,train_image,test_image)



image_dir = sys.argv[1]
train_image = sys.argv[2]
test_image = sys.argv[3]
threshold_v = int(sys.argv[4])
warnings.simplefilter("ignore")

main_fun(image_dir,train_image,test_image,threshold_v)