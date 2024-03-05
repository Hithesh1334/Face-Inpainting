import streamlit as st
from io import BytesIO
import os
import io
from streamlit_option_menu import option_menu
from PIL import Image
import cv2
import numpy as np
import requests  
from streamlit_lottie import st_lottie  
import pandas as pd
from Unet.binary_segment import binary_unet
from Gan.inpainting import inpaint_unet
from ssim import calculate_ssim
import time
import base64


def get_base64(bin_file):
    with open(bin_file,'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = "<style> .stApp{background-image: url('data:image/png;base64,%s');background-size: cover;background-repeat: no-repeat;}</style>"%bin_str  
    st.markdown(page_bg_img,unsafe_allow_html=True)

set_background("E:\\CSE\\18129294.jpg")
# set_background("E:\\CSE\\5758.jpg")
# set_background("E:\\CSE\\abstract.jpg")
# set_background("E:\\CSE\\3326663.jpg")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_hello = load_lottieurl("https://lottie.host/e0955ad6-b760-4959-831c-63af9f240283/PObNK5AnfK.json")

box = option_menu(
    menu_title=None, 
    options=["Home", "InPaint", "TrainInsight","Documentation"], 
    icons=['house', 'cloud-upload', "list-task","book"], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal")

if box=="Home":
    st.markdown("<h1 style='text-align: center; color: red;'>Wanna unmask face?</h1>", unsafe_allow_html=True)
    st_lottie(
    lottie_hello,
    speed=0.4,
    reverse=False,
    loop=True,
    quality="high", 
    height=400,
    width=None,
    key=None,
)

if box == "InPaint":
    
    menu = option_menu("Search",['Mask Inpaint','Specs Inpaint'],icons=['bi bi-person-fill','bi bi-emoji-sunglasses-fill'],orientation='horizontal')
    if menu == 'Mask Inpaint':
        mcol1,mcol2 = st.columns(2)
        with mcol1:
            image = st.file_uploader("Upload your masked image here",type=['jpg','png','jpeg'])
            
        with mcol2:
            image2 = st.file_uploader("Upload the feedback image(*optional)",type=['jpg','png','jpeg'])
            
        
        if image is not None:
                col1,col2 = st.columns(2)
                masked = Image.open(image).convert('RGB')
                print(masked,image.name,"Here")
                masked = np.array(masked)
                masked = cv2.resize(masked,(224,224))

                print(masked,image.name,"Here")
                with st.spinner("Hang In there"):
                    time.sleep(3)
                with col1:
                    st.image(masked,width=300,caption="masked photo")
                binary = binary_unet(masked,'unettest_24930.pth')
                with col2:
                    st.image(binary,width=300,caption="binary segmentation map")
                col3,col4 = st.columns(2)
                try:
                    original = "C:/Users/Hithesh Patel/Downloads/dataset/GroundTruth/"+image.name
                    original = Image.open(original).convert('RGB')
                    original = np.array(original)
                    original = cv2.resize(original,(224,224))
                    print(original,'FFSJSDKJFDLSFK')
                    with col3:
                        st.image(original,width=300,caption="Original image")
                except:
                    with col3:
                        st.image(masked,width=300,caption="Original image")
                fake = inpaint_unet(masked,binary,"inpaint3.pth")
                with col4:
                    st.image(fake,width=300,caption="Inpainted photo")
                
                fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
                upper_bound,lower_bound = 255,0 
                fake = (fake - np.min(fake)) / (np.max(fake) - np.min(fake)) * (upper_bound - lower_bound) + lower_bound
                print(fake)
                fake_path = "E:/CSE/Capstone_Project/Fakeimage/"+image.name
                cv2.imwrite(fake_path, fake)
                
        if image2 is not None:
            print("---------------------",image2)
            file_bytes = np.asarray(bytearray(image2.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            save_path = "E:/CSE/Capstone_Project/FeedbackData/"
            file_path = os.path.join(save_path, image2.name)
            cv2.imwrite(file_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            st.text("Thank you for your feedback")
    elif menu == "Specs Inpaint":
        mcol1,mcol2 = st.columns(2)
        with mcol1:
            image = st.file_uploader("Upload your masked image here",type=['jpg','png','jpeg'])
            
        with mcol2:
            image2 = st.file_uploader("Upload the feedback image(*optional)",type=['jpg','png','jpeg'])
            
        
        if image is not None:
                col1,col2 = st.columns(2)
                masked = Image.open(image).convert('RGB')
                print(masked,image.name,"Here")
                masked = np.array(masked)
                masked = cv2.resize(masked,(224,224))

                print(masked,image.name,"Here")
                with col1:
                    st.image(masked,width=300,caption="masked photo")
                binary = binary_unet(masked,"spec.pth")
                with col2:
                    st.image(binary,width=300,caption="binary segmentation map")
                col3,col4 = st.columns(2)
                try:
                    original = "C:/Users/Hithesh Patel/Downloads/dataset/GroundTruth/"+image.name
                    original = Image.open(original).convert('RGB')
                    original = np.array(original)
                    original = cv2.resize(original,(224,224))
                    print(original,'FFSJSDKJFDLSFK')
                    with col3:
                        st.image(original,width=300,caption="Original image")
                except:
                    with col3:
                        st.image(masked,width=300,caption="Original image")
                fake = inpaint_unet(masked,binary,'specsinpaint.pth')
                with col4:
                    st.image(fake,width=300,caption="Inpainted photo")
                
                fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
                upper_bound,lower_bound = 255,0 
                fake = (fake - np.min(fake)) / (np.max(fake) - np.min(fake)) * (upper_bound - lower_bound) + lower_bound
                print(fake)
                fake_path = "E:/CSE/Capstone_Project/Fakeimage/"+image.name
                cv2.imwrite(fake_path, fake)
                
        if image2 is not None:
            print("---------------------",image2)
            file_bytes = np.asarray(bytearray(image2.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            save_path = "E:/CSE/Capstone_Project/FeedbackData/"
            file_path = os.path.join(save_path, image2.name)
            cv2.imwrite(file_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            st.text("Thank you for your feedback")
        
        

elif box=="TrainInsight":
    df = pd.read_csv("E:/CSE/Capstone_Project/Files/data.csv")

    ssim_loss = df["ssim_loss"]
    gen_loss = df["gen_loss"]
    disc_whole_loss = df["mean_disc_whole_loss"]
    disc_mask_loss = df["mean_disc_whole_loss"]

    st.header("SSIM_Loss")
    average_value = ssim_loss.mean()
    min_value = ssim_loss.min()
    max_value = ssim_loss.max()
    column_name = "ssim "    
    st.text(f"max:{max_value}\nmin:{min_value}\navg:{average_value}")
    plot_data = pd.DataFrame({column_name: ssim_loss.values}, index=range(1, len(ssim_loss) + 1))
    st.line_chart(plot_data)
    
    st.header("Gen_Loss")
    average_value = gen_loss.mean()
    min_value = gen_loss.min()
    max_value = gen_loss.max()
    column_name = "gen_loss"    
    st.text(f"max:{max_value}\nmin:{min_value}\navg:{average_value}")
    plot_data = pd.DataFrame({column_name: gen_loss.values}, index=range(1, len(gen_loss) + 1))
    st.line_chart(plot_data)

    st.header("mean_dis_whole_loss")
    average_value = disc_whole_loss.mean()
    min_value = disc_whole_loss.min()
    max_value = disc_whole_loss.max()
    column_name = "gen_loss"    
    st.text(f"max:{max_value}\nmin:{min_value}\navg:{average_value}")
    plot_data = pd.DataFrame({column_name: disc_whole_loss.values}, index=range(1, len(disc_whole_loss) + 1))
    st.line_chart(plot_data)

    st.header("mean_dis_mask_loss")
    average_value = disc_mask_loss.mean()
    min_value = disc_mask_loss.min()
    max_value = disc_mask_loss.max()
    column_name = "gen_loss"    
    st.text(f"max:{max_value}\nmin:{min_value}\navg:{average_value}")
    plot_data = pd.DataFrame({column_name: disc_mask_loss.values}, index=range(1, len(disc_mask_loss) + 1))
    st.line_chart(plot_data)

elif box == "Documentation":

    image_path = []
    fake_path = "E:/CSE/Capstone_Project/Fakeimage/"
    input_path = "C:/Users/Hithesh Patel/Downloads/dataset/GroundTruth_masked/"
    original_path = "C:/Users/Hithesh Patel/Downloads/dataset/GroundTruth/"
    count = 0
    for path in os.listdir(fake_path):
        image_path.append([original_path+path,input_path+path,fake_path+path])
        count += 1
        if count > 20:
            break
    image_arr = [[] for _ in range(len(image_path))]
    for i in range(len(image_path)):
        for j in range(len(image_path[i])):
            img = Image.open(image_path[i][j]).convert('RGB')
            img = np.array(img)
            img = cv2.resize(img,(224,224))
            image_arr[i] += [img]
    print("This is image_arr",image_arr[0][0].shape)
    for i in range(len(image_arr)):
        st.image(image_arr[i],width=230, use_column_width=False,caption=["Original","input","output"])
        print(image_path[i][0],image_path[i][1])
        res = calculate_ssim(image_path[i][0],image_path[i][1])
        st.text(1-res)
        