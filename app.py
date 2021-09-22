# -*- coding: utf-8 -*-
"""FRIFD Streamlit Web Application

This script includes codes for the Basic Example of application of FRIFD 
on hand drawn images and fundus images from the DRISHTI dataset.

Example:
        $ streamlit run app.py

Dependencies :
    Follow the requirements.txt file shipped along with this repository

Todo:
    * Add few more example fundus images
    * Include baseline masks

"""

import os
import re
import time
import json
import math
import uuid
import base64
from io import BytesIO
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import SessionState

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from svgpathtools import parse_path
from streamlit_drawable_canvas import st_canvas

import scipy as sp
import scipy.linalg as la
import scipy.signal as signal

import shapely.ops as so
import shapely.geometry as sg
from shapely.geometry import Polygon

import circle_fit as cf
from ellipse import LsqEllipse
import matplotlib.image as mpimg
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

import skimage.exposure
from skimage import data
from skimage.filters import sato
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

import keras
#import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, \
    Conv2D, MaxPooling2D, ZeroPadding2D, Input, Embedding, \
    Lambda, UpSampling2D, Cropping2D, Concatenate

K.set_image_data_format('channels_first')

def main():
    st.title("FRIFD Demo")
    st.sidebar.subheader("Sections")
    session_state = SessionState.get(button_id="", color_to_label={})
    PAGES = {
        "About": about,
        "Basic Example": full_app,
        "Fundus Examples": fundus_demo,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page](session_state)

    with st.sidebar:
        st.markdown("---")
        st.markdown("Made by Aditya")

        
def about(session_state):    
    """About Page of the Application

    Includes the contents of the Application along with a GIF to demonstrate functionality.
    """
    
    st.image("./img/Logo.png")
    
    st.markdown(
        """
    Welcome to a minimalistic demo of FRI Modelling of Fourier Descriptors.
    
    On this site, you will find a Basic demo of this algorithm on Hand Drawn Images along with it's application on Fundus Images. The results can be compared to a Convolutional Neural Network in Real Time.
    
    :pencil: [Source code](https://github.com/this-is-ac/frifd)    
    """
    )
    
    st.image("./img/demo.gif")
    st.markdown(
        """
    What you can do with this Demo:

    * Test out this Algorithm on Hand Drawn Images
    * Extract the Optic Disk from a Retinal Fundus Image
    * Also has a feature to annotate Images, if needed
    """
    )


def full_app(session_state):    
    """FRIFD on Hand Drawn Images
    
    Includes a drawable canvas. Once the drawing is complete, this is passed 
    as input to the algorithm.
    """
    
    st.subheader("Basic Example")
    st.sidebar.header("Configuration")
    st.markdown(
        """
    To get started,  
    :hammer_and_wrench: : Configure canvas in the sidebar  
    :pencil2: : Use the freedraw mode to draw a custom closed object (Please use 1 stroke)  
    :joystick: : Click on the button to run the FRIFD Algorithm
    """
    )

    # Canvas parameters
    
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=250,
        drawing_mode=drawing_mode,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )
    
    if canvas_result.image_data is not None:
        with open('Image.pickle', 'wb') as handle:
            pickle.dump(canvas_result.image_data, handle)
    if canvas_result.json_data is not None:
        st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
        with open('Data.pickle', 'wb') as handle:
            pickle.dump(canvas_result.json_data["objects"], handle)

    st.subheader("Algorithm Results")
    
    width = st.sidebar.slider("Plot Width", 0.0, 10.0, 3.0, step = 0.5)
    height = st.sidebar.slider("Plot Height", 0.0, 10.0, 1.0, step = 0.5)
    K = st.sidebar.slider("Value of K", 0, 30, 3)
    
    if canvas_result.json_data["objects"]:
        algo_plot(width, height, K)   

# Custom Functions #
        
def apply_convolution(sig, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered

def FRIFD_T(x,y,K):
    N = len(x)
    n = np.arange(1,N)
    M1 = int(np.floor(N/2))
    M2 = int(np.floor(N/3))

    kT = block_ann(x[M1-M2-1:M1+M2], y[M1-M2-1:M1+M2], 2*K)

    kT = np.sort(kT+2*np.pi)
    idxT = np.where(kT>0.006)[0][0]
    T = kT[idxT]
    return T

def FRIFD_C(x,y,K,T):
    N = len(x)
    n = np.arange(1,N+1)

    ExM  = np.ones((N,1))
    for m in range(1, K):
        X = np.reshape(np.exp(1j * m * T * n), (len(np.exp(1j * m * T * n)),1))
        Y = np.reshape(np.exp(-1j * m * T * n), (len(np.exp(-1j * m * T * n)),1))
        ExM = np.hstack((X, ExM, Y))

    pa = np.dot(np.linalg.pinv(ExM), x[:])
    pb = np.dot(np.linalg.pinv(ExM), y[:])
    return [pa, pb]

def block_ann(xpy, ypy, K):
    Nn=len(xpy)
    Mm=int(np.floor(Nn/2))

    if not np.mod(Nn,2):
        xpy = xpy[:-1]
        ypy = ypy[:-1]

    U = np.zeros((2*Mm-K+1,K+1), dtype = 'complex')
    U2 = np.zeros((2*Mm-K+1,K+1), dtype = 'complex')

    for i in range(0,2*Mm-K + 1):
        U[i]=ypy[list(map(int, (np.arange(-Mm+K,-Mm-1, -1)+np.ceil(Nn/2)+(i-1)).tolist()))]
        U2[i]=xpy[list(map(int, (np.arange(-Mm+K,-Mm-1, -1)+np.ceil(Nn/2)+(i-1)).tolist()))]

    NewU = np.vstack((U, U2))
    V = SVD(NewU).T
    h = V[:, K]
    freq_est = np.angle(np.roots(h))
    freq_est[freq_est>0] = freq_est[freq_est>0]-2*np.pi
    return freq_est

def fundus_demo(session_state):
    st.subheader("Optic Disk Detection on Fundus Images")
    st.sidebar.header("Configuration")
    st.markdown(
        """
    To get started,  
    :camera: : Choose an Image from the Configuration panel
    """
    )
    
    images = {
        "Select Image  ": None,
        "Sample Image 1": './img/002.png',
        "Sample Image 2": './img/004.png',
        "Sample Image 3": './img/008.png',
    }

    plot = st.sidebar.selectbox("Select Image", list(images.keys()))
    image_name = images[plot]

    bg_image = st.sidebar.file_uploader("Custom Image:", type=["png", "jpg"])
    
    if image_name is not None or bg_image:
    
        with st.spinner('Importing Images'):
            if bg_image:
                st.image(Image.open(bg_image))
            else:
                st.image(image_name)
                image = cv2.imread(image_name)
                Points =  np.loadtxt(image_name[:-4] + "_Boundary.txt")

        st.success("Loaded Image and Boundary!")

        st.subheader("Algorithm Results")    

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

        r3 = image[:, :, 2]
        g3 = clahe.apply(image[:, :, 1])
        b3 = image[:, :, 0]

        image_enhanced = np.dstack((r3,g3,b3))
        sato_enhanced = sato(rgb2gray(image_enhanced))

        image_normff = cv2.normalize(sato_enhanced,  None, 0, 255, cv2.NORM_MINMAX)

        ret, mask = cv2.threshold(image_normff, 17, 255, cv2.THRESH_BINARY)

        mask = mask.astype(np.uint8)

        mask_inv = cv2.bitwise_not(mask)

        mask_inv = mask_inv.astype(np.uint8)

        img_bg = cv2.bitwise_and(rgb2gray(image), rgb2gray(image), mask = mask_inv)

        x = img_bg
        img_bg2 = (x-np.min(x))/(np.max(x)-np.min(x))
        img_bg2 = 255 * img_bg2
        img_bg2 = img_bg2.astype(np.uint8)

        kernel = np.ones((5,5),np.uint8)

        closing = cv2.morphologyEx(img_bg2, cv2.MORPH_CLOSE, kernel, iterations = 12)

        yoyo = closing.copy()

        hist2 = cv2.calcHist([yoyo],[0],None,[256],[0,256])

        output = apply_convolution(hist2[:, 0].T, 5)

        what = np.gradient(np.array(output))
        what  = (what - min(what)) / (max(what) - min(what))

        min_index = np.where(what==min(what))[0][0]

        local_max_index = np.where(what == max(what[min_index : min_index + 50]))[0][0]

        threshold = (min_index + local_max_index)/2

        ret, out_image = cv2.threshold(yoyo, threshold, 255, cv2.THRESH_BINARY)

        local_min_index = np.where(what == min(what[math.ceil(threshold) : 255]))[0][0]

        threshold2 = local_min_index + 30

        ret2, out_image2 = cv2.threshold(yoyo, threshold2, 255, cv2.THRESH_BINARY)

        img = out_image2

        with st.spinner('Initializing Active Contour. This step might take a while. Please be Patient 😅'):
            s = np.linspace(0, 2*np.pi, 400)
            r = 800 + 600*np.sin(s)
            c = 1070 + 600*np.cos(s)
            init = np.array([r, c]).T

            snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=10, gamma=0.001)

        st.success("Initialized Snake 🐍")

        with st.spinner("Generating Circle and Ellipse Fit Plots 📈"):
            fig, ax = plt.subplots()
            ax.imshow(rgb2gray(cv2.imread(image_name)), cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
            ax.scatter(Points[:, 1], Points[:, 0], s=0.1)
            ax.set_xticks([]) 
            ax.set_yticks([])
            ax.set_title("Active Contour")
            st.pyplot(fig)

            fig, ax = plt.subplots()
            Px = list(Points[:, 1])
            Py = list(Points[:, 0])

            X1 = list(snake[:, 1])
            X2 = list(snake[:, 0])

            data = np.array(list(zip(X2, X1)))
            xc_circle,yc_circle,radius,_ = cf.least_squares_circle((data))
            c1 = Circle((xc_circle, yc_circle), radius, color = 'black', linestyle = '--', fill = False, label = 'Circle Fit')

            lsqe = LsqEllipse()
            lsqe.fit(data)
            center, l_width, height, phi = lsqe.as_parameters()
            ellipse = Ellipse(xy=center, width=2*l_width, height=2*height, angle=np.rad2deg(phi), edgecolor='b', fc='None', lw=2, label='Ellipse Fit', zorder = 2)

            ax.add_patch(c1)
            ax.add_patch(ellipse)
            ax.imshow(rgb2gray(cv2.imread(image_name)), cmap=plt.cm.gray)
            ax.plot(Px, Py, 'o', color = 'green', markersize = 0.7, label = 'Ground Truth')
            ax.set_title("Circle and Ellipse Fits")
            ax.set_xticks([]) 
            ax.set_yticks([])
            ax.legend()
            st.pyplot(fig)
        st.success("Generated Circle and Ellipse Fits!")

        with st.spinner("Generating Intersetion"):
            p1 = Polygon(snake)
            p2 = Polygon(np.flip(Points))
            p2 = p2.buffer(2.0)

            intersection = p2.intersection(p1)

            Ratio = intersection.area / (p1.area + p2.area - intersection.area)

            box = p1.envelope

            x, y = box.exterior.coords.xy

            xs, ys = p1.exterior.xy 
            fig, ax = plt.subplots()
            ax.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            xs, ys = p2.exterior.xy    
            ax.fill(xs, ys, alpha=0.5, fc='g', ec='none')
            xs, ys = intersection.exterior.xy
            ax.plot(xs,ys)
            ax.plot(x,y)
            ax.set_title("Intersection")
            ax.set_aspect('equal', 'datalim')
            ax.set_xticks([])
            ax.set_yticks([])

            st.pyplot(fig)
            st.write("Ratio of Intersection Over Union : %", Ratio)

        st.success("Generated Intersection!")

        with st.spinner("Generating FRIFD Fit for Order 18"):
            xn = np.append(snake[:,0],snake[0,0])
            yn = np.append(snake[:,1],snake[0,1])

            xn = xn[0::6]
            yn = yn[0::6]

            xn = np.append(xn, xn[0])
            yn = np.append(yn, yn[0])

            K = 18

            T = FRIFD_T(xn,yn,K)
            [pa,pb] = FRIFD_C(xn,yn,K,T)

            Nr = np.ceil(2*np.pi/T)
            nr = np.arange(1,Nr+1)
            xr = pa[K-1]
            yr = pb[K-1]
            for m in range(1, K):
                xr = xr + pa[K+m-1]*np.exp(1j*m*nr*T) + pa[K-m-1]*np.exp(-1j*m*nr*T)
                yr = yr + pb[K+m-1]*np.exp(1j*m*nr*T) + pb[K-m-1]*np.exp(-1j*m*nr*T)

            fig, ax = plt.subplots()
            ax.imshow(rgb2gray(cv2.imread(image_name)), cmap=plt.cm.gray)
            ax.plot(xr, yr, '-r', label = "FRIFD")
            ax.scatter(Points[:, 1], Points[:, 0], s=0.1, label = "Ground Truth")
            ax.set_title("FRIFD {K=18}")
            ax.axis([0, img.shape[1], img.shape[0], 0])
            ax.set_xticks([]) 
            ax.set_yticks([])

            st.pyplot(fig)

        st.success("Generated FRIFD Fit for Order 18")
        
        st.error("The following Section will run properly, if you have a GPU. Else, use intel-tensorflow on CPU to avoid errors")
        
        with st.spinner("Generating Results from U-Net"):
            model = get_unet_light(img_rows=128, img_cols=128)
            model.load_weights('./last_checkpoint.hdf5')
            img_cnn = cv2.imread(image_name)
            img_cropped = img_cnn[int(yc_circle)-256 : int(yc_circle)+256, int(xc_circle)-256 : int(xc_circle)+256, :]
            image = cv2.resize(img_cropped, (128, 128))
            image_copy = image
            image = np.moveaxis(image, -1, 0)
            image = np.expand_dims(image, axis=0)
            
            batch_X = th_to_tf_encoding(image)
            
            batch_X = [skimage.exposure.equalize_adapthist(batch_X[i]) for i in range(len(batch_X))]
            batch_X = np.array(batch_X)
            batch_X = tf_to_th_encoding(batch_X)
            batch_X = batch_X/1.0
            
            try:
                pred = (model.predict(batch_X)[0, 0] > 0.5).astype(np.float64)
                fig, ax = plt.subplots()
                ax.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
                ax.set_title("Cropped Image")
                ax.set_xticks([])
                ax.set_yticks([])

                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.imshow(pred, cmap = 'gray')
                ax.set_title("Result from U-Net")
                ax.set_xticks([])
                ax.set_yticks([])

                st.pyplot(fig)
                st.success("Generated Result from U-Net")
            except:
                st.error("Please install intel-tensorflow to proceed.")

        st.success("All Done! 🎉")
    else:
        st.error("🚫 Please Select an Image from the Configuration Panel")
    
# Custom Functions

def tf_to_th_encoding(X):
    return np.rollaxis(X, 3, 1)

def th_to_tf_encoding(X):
    return np.rollaxis(X, 1, 4)

def get_unet_light(img_rows=256, img_cols=256):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(conv9)
    #conv10 = Flatten()(conv10)

    model = Model(inputs,conv10)

    return model

def Block_matrix(x, y):
    return np.vstack((x[:, :5], y[:, :5]))


def SVD(x):
    [A,B,C]=np.linalg.svd(x)
    return C

def Return_Delays(x):
    Kc = [0, 0, 0, 0, 1]
    Coeff = np.dot(x.T,Kc)
    Time = np.angle(np.roots(Coeff))
    return Time

def Vandermonde(row, col, Time, K):
    E = np.array([[0 for i in range(col)] for j in range(row)], dtype = 'complex')
    for i in range(row):
        for j in range(col):
            E[i][j] = np.exp(1j*(i+1)*(j-K)*Time)
    return E

def least_square(E, x, y):
    X_Output = x.reshape(-1, 1)
    Y_Output = y.reshape(-1, 1)
    a_new, na1, na2, na3 = np.linalg.lstsq(E, X_Output, rcond=-1)
    b_new, na4, na5, na6 = np.linalg.lstsq(E, Y_Output, rcond=-1)

    return a_new, b_new

def algo_plot(width, height, Kval):
    with open('Data.pickle', 'rb') as handle:
        data = pickle.load(handle)
       
    outx = []
    outy = []
    for i, path in enumerate(data):
        outx.append([])
        outy.append([])
        for x in path["path"]:
            if len(x) == 3:
                outx[i].append(x[1])
                outy[i].append(x[2])
            else:
                outx[i].append((x[1] + x[3])/2)
                outy[i].append((x[2] + x[4])/2)
    
    for i, pathy in enumerate(outy):
        value = max(pathy)
        outy[i] = [value - y for y in pathy]
    
    points_x = outx[0]
    points_y = outy[0]
    
    K = Kval
    xpy = points_x
    ypy = points_y
    xpy.append(xpy[0])
    ypy.append(ypy[0])

    xpy = np.array(xpy, dtype = 'complex')
    ypy = np.array(ypy, dtype = 'complex')
    ypy = ypy *1j

    xxx = la.toeplitz(xpy)
    yyy = la.toeplitz(ypy)
    uuu = Block_matrix(xxx, yyy)

    vvv = SVD(uuu)

    aaa = Return_Delays(vvv)

    ttt = min(aaa[aaa>0])

    eee = Vandermonde(len(xpy), 2*K+1, ttt, K)

    ann, bnn = least_square(eee, xpy, ypy)

    Middle = math.floor(len(ann)/2)
    xnn = ann[Middle]
    ynn = bnn[Middle]

    nnn = np.arange(0, len(xpy), 1)

    for k in range(K):
        xnn = xnn + ann[k+Middle+1] * np.exp(1j*(k+1)*nnn*ttt) + ann[Middle-1-k] * np.exp(-1j*(k+1)*nnn*ttt)
        ynn = ynn + bnn[k+Middle+1] * np.exp(1j*(k+1)*nnn*ttt) + bnn[Middle-1-k] * np.exp(-1j*(k+1)*nnn*ttt)
            
    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(np.real(xpy), np.imag(ypy))
    ax.plot(np.real(xnn), np.imag(ynn))
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)   

    
    return 0       
        
        
if __name__ == "__main__":
    st.set_page_config(
        page_title="FRIFD Demo", page_icon="👁"
    )
    main()
