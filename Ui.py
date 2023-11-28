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
    quality="high", 
    height=400,
    width=None,
    key=None,
)

elif box == "InPaint":
    image = st.file_uploader("Upload your masked image here",type=['jpg','png','jpeg'])
    if image is not None:
        col1,col2 = st.columns(2)
        masked = Image.open(image).convert('RGB')
        print(masked,image.name,"Here")
        masked = np.array(masked)
        masked = cv2.resize(masked,(224,224))

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
        fake_path = "E:/CSE/Capstone_Project/Fakeimage/"+image.name
        cv2.imwrite(fake_path, fake)
        try:
            files = {'image_file': open(fake_path, 'rb')}
            response = requests.post(
                "https://techhk.aoscdn.com/api/tasks/visual/scale",
                headers={'X-API-KEY': 'wxrip5cfm96ne5k5l'},
                data={'sync': '1', 'type': 'face'},
                files=files
            )

            print(response.status_code,"herererere")
            if response.status_code == 200:
                response_json = response.json()
                if response_json['data']['state'] == 1:
                    image_url = response_json['data']['image']
                    image_response = requests.get(image_url)
                    print(response.text)
                    print("hi")
                    print(image_response)
                    if image_response.status_code== 200:
                        img1 = Image.open(io.BytesIO(image_response.content))
                        print("in")
                        st.image(img1, width=300, caption="Final Image from API")
            #         else:
            #             st.image(fake,width=300,caption="API image")
            #     else:
            #         st.image(fake,width=300,caption="API image")
            # else:
            #     st.image(fake,width=300,caption="API image")
        except:
            # st.image(fake,width=300,caption="API image")
            pass
        

       
        
        

elif box=="TrainInsight":
    df = pd.read_csv("E:/CSE/Capstone_Project/Files/data.csv")

    ssim_loss = df["ssim_loss"]
    gen_loss = df["gen_loss"]
    disc_whole_loss = df["mean_disc_whole_loss"]
    disc_mask_loss = df["mean_disc_whole_loss"]

    st.header("SSIM_Loss")
    average_value = 1-ssim_loss.mean()
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
    # def read_markdown_file(file_path):
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         content = file.read()
    #     return content

    # readme_path = 'E:/CSE/Capstone_Project/readme.md'  # Replace with the actual path to your 
    # markdown_content = read_markdown_file(readme_path)
    # st.markdown(markdown_content, unsafe_allow_html=True)

    image_path = []
    fake_path = "E:/CSE/Capstone_Project/Fakeimage/"
    input_path = "C:/Users/Hithesh Patel/Downloads/capstone1/capstone/GroundTruth_masked/"
    original_path = "C:/Users/Hithesh Patel/Downloads/capstone1/capstone/GroundTruth/"
    count = 0
    for path in os.listdir(fake_path):
        image_path.append([original_path+path,input_path+path,fake_path+path])
        count += 1
        if count > 10:
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
        