import streamlit as st
from io import BytesIO
import io
from streamlit_option_menu import option_menu
from PIL import Image
import cv2
import numpy as np
import os
import imghdr
import json
import requests  
from streamlit_lottie import st_lottie  
import pandas as pd
from Unet.binary_segment import binary_unet
from Gan.inpainting import inpaint_unet
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_hello = load_lottieurl("https://lottie.host/e0955ad6-b760-4959-831c-63af9f240283/PObNK5AnfK.json")

box = option_menu(
    menu_title=None, 
    options=["Home", "Upload your Image", "Training Analysis"], 
    icons=['house', 'cloud-upload', "list-task"], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal")
def run_demo(name):
    st.image("demo_imgs/gt_imgs/"+name+".jpg",width=450,caption="Ground Truth")
    col1,col2 = st.columns(2)
    with col1:
        st.image("demo_imgs/masked_imgs/mask_"+name+".jpg",caption="Masked Photo")
    with col2:
        st.image("demo_imgs/binary_imgs/binary_"+name+".jpg",caption="Binary Segmentation Map")
    st.image("demo_imgs/pred_imgs/fake_"+name+".jpg",width=450,caption="Face-Mask Inpainted Photo")

if box=="Home":
    st.markdown("<h1 style='text-align: center; color: red;'>Wanna unmask face?</h1>", unsafe_allow_html=True)
    st_lottie(
    lottie_hello,
    speed=0.4,
    reverse=False,
    loop=True,
    quality="high", # medium ; high
    height=400,
    width=None,
    key=None,
)

elif box == "Upload your Image":
    # st.sidebar.info('Upload ***single masked person*** image . For best results  ***center the face*** in the image, and the face mask should be preferably in ***light green/blue color***.')
    image = st.file_uploader("Upload your masked image here",type=['jpg','png','jpeg'])
    if image is not None:
        col1,col2 = st.columns(2)
        masked = Image.open(image).convert('RGB')
        # original=origin
        print(masked,image.name,"Here")
        masked = np.array(masked)
        masked = cv2.resize(masked,(224,224))
        
        # original=origin
        print(masked,image.name,"Here")
        with col1:
            st.image(masked,width=300,caption="masked photo")
        binary = binary_unet(masked)
        with col2:
            st.image(binary,width=300,caption="binary segmentation map")
        col3,col4 = st.columns(2)
        try:
            original = "E:/CSE/Capstone_Project/Dataset/GroundTruth/"+image.name
            original = Image.open(original).convert('RGB')
            original = np.array(original)
            original = cv2.resize(original,(224,224))
            print(original,'FFSJSDKJFDLSFK')
            with col3:
                st.image(original,width=300,caption="Original image")
        except:
            with col3:
                st.image(masked,width=300,caption="Original image")
        fake = inpaint_unet(masked,binary)
        with col4:
            st.image(fake,width=300,caption="Inpainted photo")
        
        fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
        upper_bound,lower_bound = 255,0 
        fake = (fake - np.min(fake)) / (np.max(fake) - np.min(fake)) * (upper_bound - lower_bound) + lower_bound
        print(fake)

        fake_path = "E:/CSE/Capstone_Project/new.jpg"
        cv2.imwrite(fake_path, fake)

       
        
        

elif box=="Training Analysis":
    # import random
    # f = pd.read_csv("data.csv",header=None)
    # print("Here this is f",f.iloc[:,0], "random",len(f.iloc[:21,0]))    
    # df = pd.DataFrame(
    #     {
    #         "gen_loss":  f.iloc[1:20,0],
    #         "x":  [i for i in range(len(f.iloc[:20,0]))],
    #         # "col3": ["gen_loss","l1_loss"]
    #     }
    # )
    # st.line_chart(df,x="x",y="gen_loss")
    pass
