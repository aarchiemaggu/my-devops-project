from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import io 
from flask import Flask, send_file, jsonify, render_template, request, redirect, url_for
import cv2
import math
import uuid
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import logging
from flask import Flask, request, redirect, url_for, send_from_directory, render_template

app = Flask(name)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define upload folders
UPLOAD_FOLDER = r'c:\Users\Arshdeep\Desktop\internal\data'
OUTPUT_FOLDER = r'c:\Users\Arshdeep\Documents\SiH\output_folder'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

demo_regions = {
    "north": {
        "lon_start": 0,
        "lat_start": 0,
        "lon_end": 800,
        "lat_end": 500
    },
    "south": {
        "lon_start": 800,
        "lat_start": 500,
        "lon_end": 1600,
        "lat_end": 1200
    },
    "central": {
        "lon_start": 400,
        "lat_start": 300,
        "lon_end": 1200,
        "lat_end": 800
    }
}



# Global variables for data hiding
global shi
global exp
shi = 0
exp = 0

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# TIFF Conversion and Compression route
@app.route('/tiff-conversion', methods=['GET', 'POST'])
def tiff_conversion():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            # Save uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Output file path
            output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file.filename)[0]}_compressed.tif")

            # Process the file
            try:
                # Open the JPEG image using PIL
                with Image.open(file_path) as img:
                    img_data = np.array(img)  # Convert image to numpy array

                # Define the transform and metadata
                transform = from_origin(0, 0, 1, 1)  # Dummy transform

                # Define metadata for GeoTIFF
                profile = {
                    'driver': 'GTiff',
                    'dtype': 'uint8',
                    'count': img_data.shape[2] if len(img_data.shape) == 3 else 1,
                    'height': img_data.shape[0],
                    'width': img_data.shape[1],
                    'transform': transform,
                    'compress': 'LZW',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256
                }

                # Write the data to a compressed GeoTIFF file
                with rasterio.open(output_file, 'w', **profile) as dst:
                    if len(img_data.shape) == 3:
                        for i in range(1, img_data.shape[2] + 1):
                            dst.write(img_data[:, :, i - 1], i)
                    else:
                        dst.write(img_data, 1)

                return f'Compressed GeoTIFF saved at {output_file}'

            except Exception as e:
                return f"Error processing the file: {str(e)}"

    # If GET request, render the form
    return '''
    <h2>TIFF Conversion and Compression</h2>
    <form method="POST" enctype="multipart/form-data">
        <label for="file">Upload a JPEG file:</label>
        <input type="file" name="file">
        <input type="submit" value="Upload and Compress">
    </form>
    '''

def get_image_region(image_path, region_name):
    """
    Function to extract a specific region from the image based on predefined coordinates.
    :param image_path: Path to the uploaded image
    :param region_name: Name of the region to be cropped
    :return: BytesIO object of cropped image if successful, else None
    """
    region_meta = demo_regions.get(region_name)

    if region_meta:
        try:
            # Open the image from the specified path
            image = Image.open(image_path)

            # Crop the image based on the region's pixel coordinates
            cropped_image = image.crop((
                int(region_meta['lon_start']),
                int(region_meta['lat_start']),
                int(region_meta['lon_end']),
                int(region_meta['lat_end'])
            ))

            # Save cropped image to a buffer in-memory
            img_io = io.BytesIO()
            cropped_image.save(img_io, 'JPEG')
            img_io.seek(0)

            return img_io
        except Exception as e:
            print(f"Error cropping image: {e}")
            return None
    else:
        print(f"Region {region_name} not found in predefined data.")
        return None


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        thresholded = apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

# Data Hiding route
def PSNR(arr1, arr2):
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return np.inf
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

def find_delta(img, min_val, max_val):
    H = img.astype(dtype=np.float64)
    h, w = img.shape
    mx_capacity = 0

    for delta in range(min_val, max_val):
        map = np.zeros(w, dtype=int)
        flag = np.zeros(w, dtype=int)
        map[0] = 0  
        flag[0] = 1

        for i in range(1, len(map)):
            map[i] = (map[i - 1] + delta) % w  
            if flag[map[i]] == 1:
                map[i] = (map[i] + 1) % w
            flag[map[i]] = 1

        diff = np.zeros((h, w // 2), dtype=np.float64)
        for i in range(w // 2):
            x = 2 * i
            y = x + 1
            diff[:, i] = H[:, map[x]] - H[:, map[y]]

        u_val, counts = np.unique(diff, return_counts=True)
        if len(u_val) == 1:
            peak = u_val[0]
            capacity = w * h // 2
        else:
            ind = np.argmax(counts)
            capacity = counts[ind]
            peak = u_val[ind]

        if capacity > mx_capacity:
            mx_capacity = capacity
            mx_delta = delta
            mx_map = map
            mx_peak = peak
            diff_img = diff

    return mx_delta, mx_capacity, mx_map, mx_peak, diff_img

def histogram_shifting(arr, peak):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] > peak:
                arr[i][j] += 1
    return arr

def embedding(arr, peak, data):
    ind = 0
    for i in range(len(arr)):
        if ind == len(data):
            break
        for j in range(len(arr[0])):
            if arr[i][j] == peak:
                if data[ind] == '1':
                    arr[i][j] += 1
                    ind += 1
                    if ind == len(data):
                        break
                else:
                    ind += 1
                    if ind == len(data):
                        break
    return arr

def generate_transformed_embedded(transformed_image, embedded_diff_image, embedd=False):
    n = transformed_image.shape[1]
    half_n = n // 2
    transformed_with_embedded = np.copy(transformed_image)
    for i in range(half_n):
        transformed_with_embedded[:, 2 * i] = transformed_image[:, 2 * i + 1] + embedded_diff_image[:, i]
        if embedd:
            global shi
            shi += embedded_diff_image.shape[0]
    return transformed_with_embedded

def construct_image(transformed_with_embedded, map):
    sorted_columns = np.argsort(map)
    marked_image = transformed_with_embedded[:, sorted_columns]
    return marked_image

@app.route('/data-hiding', methods=['GET', 'POST'])
def data_hiding():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index1.html', output={"embedding_successful": False, "psnr_value": "Error", "relative_capacity": "Error"})

        file = request.files['image']
        if file.filename == '':
            return render_template('index1.html', output={"embedding_successful": False, "psnr_value": "Error", "relative_capacity": "Error"})

        secret_code = request.form.get('secret_code', '')

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load the image and perform embedding
        img = Image.open(file_path)
        grayscale_image = img.convert("L")
        image = np.array(grayscale_image)

        # Find delta, capacity, map, peak, and difference image
        mx_delta, mx_capacity, mx_map, mx_peak, difference_image = find_delta(image, 1, image.shape[1])

        # Transform and embed the data
        transformed_image = image[:, mx_map]
        inter_diff_image = histogram_shifting(np.copy(difference_image), mx_peak)
        marked_diff_image = embedding(np.copy(inter_diff_image), mx_peak, secret_code)

        # Embed data into transformed image
        transformed_with_embedded = generate_transformed_embedded(transformed_image, marked_diff_image, embedd=True)

        # Construct marked image
        marked_image = construct_image(transformed_with_embedded, mx_map)

        def is_embedding_successful(original_image, marked_image, threshold=30):
            psnr_value = PSNR(original_image, marked_image)
            return psnr_value > threshold, psnr_value

        embedding_successful, psnr_value = is_embedding_successful(image, marked_image)

        marked_image_pil = Image.fromarray(marked_image.astype(np.uint8))
        
        output_path = os.path.join(UPLOAD_FOLDER, "embedded_image.png")
        marked_image_pil.save(output_path)

        total_pixels = image.shape[0] * image.shape[1]
        relative_capacity = mx_capacity / total_pixels

        # Prepare output data for rendering in HTML
        output_data = {
            "embedding_successful": embedding_successful,
            "psnr_value": psnr_value,
            "relative_capacity": relative_capacity
        }

        return render_template('index1.html', output=output_data)

    # If GET request, render the form
    return '''
    <h2>Data Hiding</h2>
    <form method="POST" enctype="multipart/form-data">
        <label for="image">Upload an image:</label>
        <input type="file" name="image" required>
        <br>
        <label for="secret_code">Secret code:</label>
        <input type="text" name="secret_code" required>
        <input type="submit" value="Hide Data">
    </form>
    '''

@app.route('/extract-region', methods=['GET', 'POST'])
def upload_image():
    """
    Flask route to upload an image and choose a region to crop.
    :return: HTML template for image upload and cropping options.
    """
    if request.method == 'POST':
        uploaded_file = request.files.get('image')
        region_name = request.form.get('region')

        if uploaded_file and region_name in demo_regions:
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)  # Make sure to use UPLOAD_FOLDER
            uploaded_file.save(image_path)

            # Get the cropped image region
            image_data = get_image_region(image_path, region_name)
            if image_data:
                return send_file(image_data, mimetype='image/jpeg', as_attachment=True, download_name=f'{region_name}_region.jpg')
            else:
                return jsonify({"error": "Error processing image."}), 500

    return render_template('upload.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'dehazed/'

# Ensure upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Apply mask function
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

# Apply threshold function
def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

# Simplest color balance function
def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert 0 < percent < 100

    half_percent = percent / 200.0
    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        height, width = channel.shape
        flat = channel.flatten()
        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        thresholded = apply_threshold(channel, low_val, high_val)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

# Route to handle the upload and dehazing
@app.route('/dehazing', methods=['GET', 'POST'])
def upload_dehazing_image():
    if request.method == 'GET':
        # Render a simple HTML form to upload an image
        return '''
        <h2>Upload an image for dehazing</h2>
        <form method="POST" enctype="multipart/form-data">
            <label for="file">Choose an image (png/jpg/jpeg):</label>
            <input type="file" name="file" accept="image/png, image/jpeg" required>
            <input type="submit" value="Upload">
        </form>
        '''

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and process the image using OpenCV
            img = cv2.imread(filepath)
            if img is None:
                return "Error: Could not read the image. Please check the file format."

            # Apply the dehazing function
            processed_img = simplest_cb(img, 1)

            # Save the dehazed image with the same name as the original in the 'dehazed' folder
            processed_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

            if not cv2.imwrite(processed_path, processed_img):
                return "Error: Failed to save the processed image."

            # Redirect to a route to display the processed image along with the original
            return redirect(url_for('show_images', original=filename, processed=filename))

# Route to display the original and processed images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dehazed/<filename>')
def dehazed_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/show_images')
def show_images():
    original_filename = request.args.get('original')
    processed_filename = request.args.get('processed')
    return f'''
    <h2>Original Image</h2>
    <img src="{url_for('uploaded_file', filename=original_filename)}" alt="Original Image" width="300">
    <h2>Dehazed Image</h2>
    <img src="{url_for('dehazed_file', filename=processed_filename)}" alt="Dehazed Image" width="300">
    '''



# Route to serve the processed image
@app.route('/processed/<filename>')
def send_processed_image(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions (detection logic)
def detect_flood(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    water_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    flood_pixels = np.sum(water_mask == 255)
    total_pixels = water_mask.size
    flood_percentage = (flood_pixels / total_pixels) * 100
    flood_threshold = 10.0

    if flood_percentage > flood_threshold:
        return f"Flood detected! ({flood_percentage:.2f}% of the image shows flood-like areas)"
    else:
        return f"No flood detected. ({flood_percentage:.2f}% of the image shows flood-like areas)"

def detect_cyclone(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 200])
    upper_gray = np.array([180, 50, 255])
    cloud_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    cyclone_pixels = np.sum(cloud_mask == 255)
    total_pixels = cloud_mask.size
    cyclone_percentage = (cyclone_pixels / total_pixels) * 100
    cyclone_threshold = 15.0

    if cyclone_percentage > cyclone_threshold:
        return f"Cyclone detected! ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)"
    else:
        return f"No cyclone detected. ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)"

def detect_forest_fire(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    fire_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    fire_pixels = np.sum(fire_mask == 255)
    total_pixels = fire_mask.size
    fire_percentage = (fire_pixels / total_pixels) * 100
    fire_threshold = 5.0

    if fire_percentage > fire_threshold:
        return f"Forest fire detected! ({fire_percentage:.2f}% of the image shows fire-like areas)"
    else:
        return f"No forest fire detected. ({fire_percentage:.2f}% of the image shows fire-like areas)"

# Detect route
@app.route('/detection', methods=['POST'])
def detect():
    logging.info('Received request at /detect')
    image_file = request.files.get('image', None)

    if image_file and image_file.filename != '':
        logging.info(f"Image file received: {image_file.filename}")

        # Ensure 'static' directory exists
        if not os.path.exists('static'):
            os.makedirs('static')

        # Save the image to the 'static' directory
        image_path = os.path.join('static', image_file.filename)
        logging.info(f"Saving image to {image_path}")
        image_file.save(image_path)

        # Run the detection functions
        flood_result = detect_flood(image_path)
        cyclone_result = detect_cyclone(image_path)
        fire_result = detect_forest_fire(image_path)

        # Render results on result2.html
        return render_template('result2.html', 
                               image_url=image_path, 
                               flood_result=flood_result, 
                               cyclone_result=cyclone_result, 
                               fire_result=fire_result)
    else:
        logging.error("No image uploaded or filename is empty")
        return "No image uploaded!", 400
    
OUTPUT_DIR = 'output'

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to adjust contrast
def adjust_contrast(image, contrast_factor):
    """Adjust the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)

@app.route('/adjust-contrast', methods=['GET', 'POST'])
def upload_and_adjust_contrast():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template('index4.html', error="No image part in the request")

        file = request.files['image']
        if file.filename == '':
            return render_template('index4.html', error="No selected file")

        # Get the contrast factor from the form (default is 1.0)
        contrast_factor = float(request.form.get('contrast', 1.0))

        # Open the uploaded image
        img = Image.open(file)

        # Adjust the contrast
        img = adjust_contrast(img, contrast_factor)
        img = img.resize((350, 350), Image.Resampling.LANCZOS)

        # Save the image to the output folder with a modified filename
        output_filename = f"contrasted_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        img.save(output_path, 'JPEG')

        # Render the template with the adjusted image
        return render_template('index4.html', output_image=output_filename, message="Image saved successfully")

    return render_template('index4.html')

# Route to serve the images from the output folder
@app.route('/output/<filename>')
def send_output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# Configuration for uploads (optional, in case needed later)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Set the upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_flood(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for water
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    water_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    flood_pixels = np.sum(water_mask == 255)
    total_pixels = water_mask.size
    flood_percentage = (flood_pixels / total_pixels) * 100

    flood_threshold = 10.0  # Threshold for flood detection
    if flood_percentage > flood_threshold:
        return f"Flood detected! ({flood_percentage:.2f}% of the image shows flood-like areas)"
    else:
        return f"No flood detected. ({flood_percentage:.2f}% of the image shows flood-like areas)"

def detect_cyclone(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for clouds
    lower_gray = np.array([0, 0, 200])
    upper_gray = np.array([180, 50, 255])
    cloud_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    cyclone_pixels = np.sum(cloud_mask == 255)
    total_pixels = cloud_mask.size
    cyclone_percentage = (cyclone_pixels / total_pixels) * 100

    cyclone_threshold = 15.0  # Threshold for cyclone detection
    if cyclone_percentage > cyclone_threshold:
        return f"Cyclone detected! ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)"
    else:
        return f"No cyclone detected. ({cyclone_percentage:.2f}% of the image shows cyclone-like areas)"

def detect_forest_fire(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Image not found!"

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for fire
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    fire_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    fire_pixels = np.sum(fire_mask == 255)
    total_pixels = fire_mask.size
    fire_percentage = (fire_pixels / total_pixels) * 100

    fire_threshold = 5.0  # Threshold for fire detection
    if fire_percentage > fire_threshold:
        return f"Forest fire detected! ({fire_percentage:.2f}% of the image shows fire-like areas)"
    else:
        return f"No forest fire detected. ({fire_percentage:.2f}% of the image shows fire-like areas)"

@app.route('/cal', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Save the file to a temporary location
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Perform detection
        flood_result = detect_flood(image_path)
        cyclone_result = detect_cyclone(image_path)
        fire_result = detect_forest_fire(image_path)

        return render_template('result3.html', flood_result=flood_result, 
                               cyclone_result=cyclone_result, 
                               fire_result=fire_result)

    return render_template('index5.html')

if name == 'main':
    app.run(debug=True)