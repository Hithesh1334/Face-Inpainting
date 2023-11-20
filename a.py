import requests
from PIL import Image
from io import BytesIO

# Make a POST request to the API endpoint
response = requests.post(
    "https://techhk.aoscdn.com/api/tasks/visual/scale",
    headers={'X-API-KEY': 'wx5s7tg963m9im2t7'},
    data={'sync': '1', 'type': 'face'},
    files={'image_file': open('E:/CSE/Capstone_Project/new.jpg', 'rb')}
)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the response JSON
    response_json = response.json()

    # Check if the task is complete
    if response_json['data']['state'] == 1:
        # Get the image URL from the response
        image_url = response_json['data']['image']

        # Make a GET request to the image URL
        image_response = requests.get(image_url)

        # Check if the image request was successful
        if image_response.status_code == 200:
            # Open the image using PIL
            img = Image.open(BytesIO(image_response.content))
            print(img)
            # Display the image
            img.show()

        else:
            print(f"Failed to retrieve image. Status code: {image_response.status_code}")

    else:
        print(f"Task is not complete. State detail: {response_json['data']['state_detail']}")

else:
    print(f"Request failed. Status code: {response.status_code}")
