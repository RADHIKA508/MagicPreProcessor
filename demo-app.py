import cv2
import streamlit as st
from PIL import Image
import numpy as np

def grayscale_image(image):
    # Convert the image to grayscale
    if len(image.shape) != 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image, size_of_kernel = 5, sigma = 0):
    blurred_img = cv2.GaussianBlur(image, (size_of_kernel, size_of_kernel), sigma)
    return blurred_img

def sobel_filter(image, size_of_kernel = 3):
        # Apply the Sobel filter to the grayscale image
    sobel_x = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 1, 0, ksize=size_of_kernel)
    sobel_y = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 0, 1, ksize=size_of_kernel)

    # Combine the x and y Sobel gradients
    sobel_combined = cv2.addWeighted(np.absolute(sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0)

    # Normalize the Sobel output for better visualization
    sobel_output = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return sobel_output

def non_maximal_suppression(image, size_of_kernel):
    gradient_x = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 1, 0, ksize=size_of_kernel)
    gradient_y = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 0, 1, ksize=size_of_kernel)
    gradient_magnitude = cv2.convertScaleAbs(np.sqrt(gradient_x**2 + gradient_y**2))
    gradient_angle = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

    edges = gradient_magnitude
    angles = gradient_angle

    suppressed_edges = edges.copy()
    height, width = edges.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = angles[i, j]

            # Check the gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
                prev_pixel = edges[i, j - 1]
                next_pixel = edges[i, j + 1]
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                prev_pixel = edges[i - 1, j + 1]
                next_pixel = edges[i + 1, j - 1]
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                prev_pixel = edges[i - 1, j]
                next_pixel = edges[i + 1, j]
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                prev_pixel = edges[i - 1, j - 1]
                next_pixel = edges[i + 1, j + 1]

            # Suppress non-maximum values
            if edges[i, j] < prev_pixel or edges[i, j] < next_pixel:
                suppressed_edges[i, j] = 0

    return suppressed_edges

def double_threshold(image, threshold_low, threshold_high):

    output_image = np.zeros_like(image)
    # Set strong edges
    output_image[image >= threshold_high] = 255
    # Set weak edges
    output_image[(image >= threshold_low) & (image < threshold_high)] = 100

    return output_image

def hysteresis_threshold(image,double_threshold_image, threshold_low, threshold_high):
    strong_edges = cv2.dilate(double_threshold_image, None)
    weak_edges = (image >= threshold_low) & (image < threshold_high)
    connected_edges = cv2.connectedComponents(weak_edges.astype(np.uint8))[1]
    double_threshold_image[(connected_edges > 0) & (strong_edges > 0)] = 255

    return double_threshold_image

import_modules_code = """
import cv2\n
import streamlit as st\n
from PIL import Image\n
import numpy as np\n

"""

grayscale_image_code = """
def grayscale_image(image):\n
    if len(image.shape) != 3:\n
        return image\n
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)\n

"""
gaussian_blur_code = """
def gaussian_blur(image, size_of_kernel = 5, sigma = 0):\n
    blurred_img = cv2.GaussianBlur(image, (size_of_kernel, size_of_kernel), sigma)\n
    return blurred_img\n

"""
sobel_filter_code = """
def sobel_filter(image, size_of_kernel = 3):\n
    sobel_x = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 1, 0, ksize=size_of_kernel)\n
    sobel_y = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 0, 1, ksize=size_of_kernel)\n

    sobel_combined = cv2.addWeighted(np.absolute(sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0)\n

    sobel_output = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)\n

    return sobel_output\n
"""
non_maximal_suppression_code = """
def non_maximal_suppression(image, size_of_kernel):\n
    gradient_x = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 1, 0, ksize=size_of_kernel)\n
    gradient_y = cv2.Sobel(grayscale_image(image), cv2.CV_64F, 0, 1, ksize=size_of_kernel)\n
    gradient_magnitude = cv2.convertScaleAbs(np.sqrt(gradient_x**2 + gradient_y**2))\n
    gradient_angle = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)\n

    edges = gradient_magnitude\n
    angles = gradient_angle\n

    suppressed_edges = edges.copy()\n
    height, width = edges.shape\n

    for i in range(1, height - 1):\n
        for j in range(1, width - 1):\n
            angle = angles[i, j]\n

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):\n
                prev_pixel = edges[i, j - 1]\n
                next_pixel = edges[i, j + 1]\n
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):\n
                prev_pixel = edges[i - 1, j + 1]\n
                next_pixel = edges[i + 1, j - 1]\n
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):\n
                prev_pixel = edges[i - 1, j]\n
                next_pixel = edges[i + 1, j]\n
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):\n
                prev_pixel = edges[i - 1, j - 1]\n
                next_pixel = edges[i + 1, j + 1]\n

            if edges[i, j] < prev_pixel or edges[i, j] < next_pixel:\n
                suppressed_edges[i, j] = 0\n

    return suppressed_edges\n

"""
double_threshold_code = """
def double_threshold(image, threshold_low, threshold_high):\n

    output_image = np.zeros_like(image)\n
    output_image[image >= threshold_high] = 255\n
    output_image[(image >= threshold_low) & (image < threshold_high)] = 100\n

    return output_image\n

"""
hysteresis_threshold_code = """
def hysteresis_threshold(image,double_threshold_image, threshold_low, threshold_high):\n
    strong_edges = cv2.dilate(double_threshold_image, None)\n
    weak_edges = (image >= threshold_low) & (image < threshold_high)\n
    connected_edges = cv2.connectedComponents(weak_edges.astype(np.uint8))[1]\n
    double_threshold_image[(connected_edges > 0) & (strong_edges > 0)] = 255\n

    return double_threshold_image\n

"""

# Streamlit app code
st.title("Magic OpenCV")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

def main():
    final_code_output = ""
    code_functions_output = import_modules_code
    main_code_output = ""
    processed_image_count = 2

    if uploaded_file is None:
        return None
    # Read the image using PIL
    image = Image.open(uploaded_file)

    # Convert the image to OpenCV format
    img_array = np.array(image.convert('RGB'))

    # Display the uploaded image
    st.image(image, caption="Original Image", use_column_width=True)

    # Checkboxes
    grayscale_checkbox = st.checkbox("GRAYSCALE")
    gaussian_blur_checkbox = st.checkbox("GAUSSIAN BLUR")
    sobel_filter_checkbox = st.checkbox("SOBEL FILTER")
    double_threshold_checkbox = st.checkbox("DOUBLE THRESHOLDING")

    if grayscale_checkbox:
        # Convert and display the grayscale image
        img_array = grayscale_image(img_array)

        code_functions_output += grayscale_image_code
        main_code_output += "\nimg" + str(processed_image_count) + " = " +\
        "grayscale_image(img" + str(processed_image_count - 1) + ")\n" 
        processed_image_count += 1

    if gaussian_blur_checkbox:
        size_of_kernel = st.slider("Size of Kernel", 1, 100, step = 2, key = "gaussian")
        sigma = st.slider("Sigma for gaussian",0,100,key = "sigma")
        img_array = gaussian_blur(img_array, size_of_kernel, sigma)
        
        code_functions_output += gaussian_blur_code
        main_code_output += "\nimg" + str(processed_image_count) + " = " +\
        "gaussian_blur(img" + str(processed_image_count - 1) + ", " +\
        str(size_of_kernel) + ", " + str(sigma) + ")\n"
        processed_image_count += 1


    if sobel_filter_checkbox:
        size_of_kernel = st.slider("Size of Kernel for sobel", 1, 100, step = 2, key = "sobel")
        img_array = sobel_filter(img_array, size_of_kernel)

        code_functions_output += sobel_filter_code
        main_code_output += "\nimg" + str(processed_image_count) + " = " +\
        "sobel_filter(img" + str(processed_image_count - 1) + ", " +\
        str(size_of_kernel) + " )\n"
        processed_image_count += 1

        non_maximal_suppression_checkbox = st.checkbox("NON MAXIMAL SUPPRESSION")
        if non_maximal_suppression_checkbox:
            img_array = non_maximal_suppression(img_array, size_of_kernel)
            
            code_functions_output += non_maximal_suppression_code
            main_code_output += "\nimg" + str(processed_image_count) + " = " +\
            "non_maximal_suppression(img" + str(processed_image_count - 1) + ", " +\
            str(size_of_kernel) + " )\n"
            processed_image_count += 1
        
    if double_threshold_checkbox:
        threshold_low = st.slider("Low Threshold",0, 250 ,key = "threshold_low")
        threshold_high = st.slider("High Threshold",0, 250, key = "threshold high")
        org_img = img_array.copy()
        img_array = double_threshold(img_array, threshold_low, threshold_high)
        
        code_functions_output += double_threshold_code
        main_code_output += "\nimg" + str(processed_image_count) + " = " +\
        "double_threshold(img" + str(processed_image_count - 1) + ", " +\
        str(threshold_low) + ", " + str(threshold_high) + ")\n"
        processed_image_count += 1

        hysteresis_threshold_checkbox = st.checkbox("HYSTERESIS THRESHOLDING")
        
        if hysteresis_threshold_checkbox:
            img_array = hysteresis_threshold(org_img, img_array, threshold_low, threshold_high)

            code_functions_output += hysteresis_threshold_code
            main_code_output += "\nimg" + str(processed_image_count) + " = " +\
            "hysteresis_threshold(img" + str(processed_image_count - 2) + ", " +\
            "img" + str(processed_image_count - 1) + ", " + str(threshold_low) + ", " \
                + str(threshold_high) + ")\n"
            processed_image_count += 1

    st.image(img_array, caption = "Processed Image", channels="GRAY",\
              use_column_width=True)
    
    code_functions_output += """
    \nimg1 = cv2.imread(<your image>)\n
    """

    main_code_output += "\ncv2.imshow( <your caption>, " + "img" + str(processed_image_count-1) + ")\n"

    return code_functions_output + main_code_output 

code_output = main()

if st.button("GENERATE CODE"):
    st.write(code_output + "\ncv2.waitKey(0)")