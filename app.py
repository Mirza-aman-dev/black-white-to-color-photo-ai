from flask import Flask, render_template, request, send_file, redirect, url_for
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Paths to load the pre-trained colorization model
PROTOTXT = os.path.join("colorization_deploy_v2.prototxt")
POINTS = os.path.join("pts_in_hull.npy")
MODEL = os.path.join("colorization_release_v2.caffemodel")

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to colorize the image
def colorize_image(image_path):
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Save the colorized image
    output_path = os.path.join("static", "colorized_image.jpg")
    cv2.imwrite(output_path, colorized)

    return output_path

# Home route to upload the image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file is selected
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join("uploads", filename)
            file.save(filepath)

            # Colorize the image
            output_path = colorize_image(filepath)

            # Redirect to the result page
            return redirect(url_for('result', image_path=output_path))
    return render_template('upload.html')

# Result route to display the colorized image and provide download option
@app.route('/result')
def result():
    image_path = request.args.get('image_path', None)
    if image_path:
        return render_template('result.html', image_path=image_path)
    return "No image found"

# Download the colorized image
@app.route('/download/<filename>')
def download_image(filename):
    return send_file(os.path.join("static", filename), as_attachment=True)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
