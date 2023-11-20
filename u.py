# wx5s7tg963m9im2t7 api key pic wish
from io import BytesIO
import io
import streamlit as st
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
# elif box == "Documentation":
#     st.markdown("<h1 style='text-align: center; color: white;'>Face Mask Inpainting</h1>", unsafe_allow_html=True)
#     col1,col2,col3 = st.columns(3)
#     with col1:
#         st.image("info_imgs/test1.jpg")
#         st.image("info_imgs/test2.jpg")
#     with col2:
#         st.markdown(" $~$\n\n$~$\n\n $~~~~~~~~~~~~~~$ ---------->")
#         st.markdown(" $~$\n\n$~$\n\n$~$\n\n$~$\n\n$~$\n\n$~~~~~~~~~~~~~~$ ----------->")
#     with col3:
#         st.image("info_imgs/test1.gif")
#         st.image("info_imgs/test2.gif")
    
#     st.markdown("This project attempted to achieve the paper **[A novel GAN-based network for unmasking of "
#                 "masked face](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9019697)**. The model "
#                 "is designed to remove the face-mask from facial image and inpaint the left-behind region based "
#                 "on a novel GAN-network approach. ")
#     st.image("info_imgs/md_archi.png")
#     st.markdown("Rather than using the traditional pix2pix U-Net method, in this work the model consists of two main modules, "
#                 "map module and editing module. In the first module, we detect the face-mask object and generate a "
#                 "binary segmentation map for data augmentation. In the second module, we train the modified U-Net "
#                 "with two discriminators using masked image and binary segmentation map.")
#     st.markdown("***Feel free to play it around:***")
#     st.markdown(":point_left: To get started, you can choose ***Demo Image*** to see the performance of the model.")
#     st.markdown(":camera: Feel free to ***upload*** any masked image you want and see the performance.")
#     st.markdown(":chart_with_upwards_trend: Also, press ***Training Analysis*** to see the training insight.")


# elif box == "Demo Image":
#     st.sidebar.write("---")

#     demoimg_dir = "demo_imgs/gt_imgs"
#     photos=[]
#     for file in os.listdir(demoimg_dir):
#         filepath = os.path.join(demoimg_dir,file)
#         if imghdr.what(filepath) is not None:
#             photos.append(file[:-4])
#     photos.sort()

#     inpaint_option = st.sidebar.selectbox("Please select a sample image, then click the 'Inpaint!' button.",photos)
#     inpaint = st.sidebar.button("Inpaint !")

#     if inpaint:
#         st.empty()
#         run_demo(inpaint_option)


elif box == "Upload your Image":
    # st.sidebar.info('Upload ***single masked person*** image . For best results  ***center the face*** in the image, and the face mask should be preferably in ***light green/blue color***.')
    image = st.file_uploader("Upload your masked image here", type=['jpg', 'png', 'jpeg'])
    if image is not None:
        col1, col2 = st.columns(2)
        masked = Image.open(image).convert('RGB')
        masked = np.array(masked)
        masked = cv2.resize(masked, (224, 224))

        with col1:
            st.image(masked, width=300, caption="masked photo")

        binary = binary_unet(masked)

        with col2:
            st.image(binary, width=300, caption="binary segmentation map")

        col3, col4 = st.columns(2)

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
            st.image(fake, width=300, caption="Inpainted photo")

        # Save inpainted photo
        fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)
        upper_bound,lower_bound = 255,0 
        fake = (fake - np.min(fake)) / (np.max(fake) - np.min(fake)) * (upper_bound - lower_bound) + lower_bound
        print(fake)
        fake_path = "E:/CSE/Capstone_Project/new.jpg"
        cv2.imwrite(fake_path, fake)

        # Prepare the inpainted photo for API request
        files = {'image_file': open(fake_path, 'rb')}

        # # Make a POST request to the API endpoint
        response = requests.post(
            "https://techhk.aoscdn.com/api/tasks/visual/scale",
            headers={'X-API-KEY': 'wx5s7tg963m9im2t7'},
            data={'sync': '1', 'type': 'face'},
            files=files
        )

        # # Check if the request was successful (status code 200)
        
        print(response.status_code,"herererere")
        if response.status_code == 200:

            # Parse the response JSON
            response_json = response.json()

            # Check if the task is complete
            if response_json['data']['state'] == 1:
                # Get the image URL from the response
                image_url = response_json['data']['image']

                # Make a GET request to the image URL
                image_response = requests.get(image_url)
                print(response.text)
                print("hi")
                print(image_response)
                # Check if the image request was successful
                if image_response.status_code== 200:
                    # Open the image using PIL
                    img1 = Image.open(io.BytesIO(image_response.content))
                    # img1.show()
                    print("in")
                    # Display the image
                    st.image(img1, width=300, caption="Final Image from API")

                else:
                    st.error(f"Failed to retrieve image from API. Status code: {image_response.status_code}")

            else:
                st.error(f"Task is not complete. State detail: {response_json['data']['state_detail']}")

        else:
            st.error(f"API request failed. Status code: {response.status_code}")
        
        

elif box=="Training Analysis":
    fid_frames = []
    f = pd.read_csv("Book1.csv",header=None)
    fid_frames.append(f)
    # for i in range (1,17):
        # f = pd.read_csv("Book1"+str(i),header=None)
        # fid_frames.append(f)
    df = pd.concat(fid_frames)
    dffid = pd.DataFrame(columns=['gen loss','discriminator loss'])
    for i in range(len(df)-1):
        print(i,len(df))
        if i%2==0:
            new_row = {"gen loss":df.iloc[i].values[0],"discriminator loss":df.iloc[i+1].values[0]}
            dffid = dffid.append(new_row,ignore_index=True)
    dffid.set_index("gen loss",inplace=True)
    st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Frechet Inception Distance (FID)***")
    st.line_chart(dffid)


    # losses_frames = []
    # df_header = pd.read_csv("metrics_20k/loss_epoch1",header=None)[0:1]
    # df_header = df_header.to_string()[66:]
    # df_loss = pd.DataFrame(columns=(df_header.split('   ')))
    # df_loss.insert(0,'gen loss','') 
    # for i in range (1,17):
    #     f = pd.read_csv("metrics_20k/loss_epoch"+str(i),header=None)[1:]
    #     losses_frames.append(f)
    # df_l = pd.concat(losses_frames)

    # for i in range(len(df_l)):
    #     if i%2==0:
    #         new_row = {"gen loss":float(df.iloc[i].values[0])}
    #     if i%2!=0:
    #         loss_terms = df_l.iloc[i].values[0].split('    ')
    #         new_row["gen"]=float(loss_terms[0])
    #         new_row["disc_whole"]=float(loss_terms[1])
    #         new_row["disc_mask"]=float(loss_terms[2])
    #         new_row["l1_loss"]=float(loss_terms[3])
    #         new_row["ssim_loss"]=float(loss_terms[4])
    #         new_row["percep"]=float(loss_terms[5])
    #         df_loss = df_loss.append(new_row,ignore_index=True)
    # df_loss.set_index("gen loss",inplace=True)

    # dfgen = df_loss[["gen"]]
    # st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Generator Loss***")
    # st.line_chart(dfgen)

    # dfdisc = df_loss[["disc_whole","disc_mask"]]
    # st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***Discriminators Loss***")
    # st.line_chart(dfdisc)

    # dflosses = df_loss[["l1_loss","ssim_loss","percep"]]
    # st.write("$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$***L1,SSIM,Perceptual Loss***")
    # st.line_chart(dflosses)