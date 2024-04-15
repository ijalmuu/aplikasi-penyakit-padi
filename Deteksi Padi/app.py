import os
import predict_model
from flask import Flask, render_template, request, make_response, request, redirect, url_for, session, Response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
from werkzeug.utils import secure_filename
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from flask_bcrypt import Bcrypt
import uuid
from datetime import datetime
import cv2
import json

app = Flask(__name__)
app.secret_key = "jiqowdjio1d19"
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Bcrypt untuk enkripsi password
bcrypt = Bcrypt(app)

# Initialize Firestore DB
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'deteksi-padi.appspot.com'})
db = firestore.client()

# Initialize Firebase Storage
bucket = storage.bucket()

def generate_frames_web(path_x):
    yolo_output = predict_model.video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)

@app.route("/", methods=['GET', 'POST'])
@app.route("/login", methods=['GET', 'POST'])
@nocache
def login():
    if request.method == 'GET':
        session['detection_done'] = False
        return render_template("login.html")
    elif request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Cari pengguna dengan username yang cocok
        user_ref = db.collection('users').where('username', '==', username).limit(1).get()
        if user_ref:
            user_data = user_ref[0].to_dict()
            nama_user = user_data.get('nama')
            username = user_data.get('username')
            hashed_password = user_data.get('password')

            # Periksa apakah password yang dimasukkan sesuai dengan password yang dienkripsi
            if bcrypt.check_password_hash(hashed_password, password):
                session['nama_user'] = nama_user
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                return render_template("login.html", error="Password salah. Silakan coba lagi.")
        else:
            return render_template("login.html", error="Username tidak ditemukan.")

@app.route("/register", methods=['GET', 'POST'])
@nocache
def register():
    if request.method == 'GET':
        return render_template("register.html")
    elif request.method == 'POST':
        nama = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')

        # Cek apakah username sudah ada atau belum
        user_ref = db.collection('users').where('username', '==', username).get()
        if user_ref:
            return render_template("register.html", error="Username sudah digunakan. Silakan gunakan username lain.")

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Menyimpan data user ke dalam database
        user_ref = db.collection('users').document()
        user_ref.set({
            'nama': nama,
            'username': username,
            'password': hashed_password
        })

        return redirect(url_for('login'))

@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/detection-yolo", methods=["POST"])
@nocache
def detection_yolo():
    counts, title = predict_model.detection_yolo()
    print(counts)

    # Menghasilkan nama acak untuk gambar
    img_normal_random_name = str(uuid.uuid4()) + ".jpg"
    img_now_random_name = str(uuid.uuid4()) + ".jpg"

    # Mengunggah gambar yang akan dideteksi
    img_normal_path = "static/img/img_normal.jpg"
    img_normal_blob = bucket.blob(img_normal_random_name)
    img_normal_blob.upload_from_filename(img_normal_path)

    # Mengubah objek blob menjadi publik
    img_normal_blob.make_public()

    # Mendapatkan URL gambar yang diunggah secara publik
    img_normal_public_url = img_normal_blob.public_url

    # Mengunggah gambar hasil deteksi
    img_now_path = "static/img/img_now.jpg"
    img_now_blob = bucket.blob(img_now_random_name)
    img_now_blob.upload_from_filename(img_now_path)

    # Mengubah objek blob menjadi publik
    img_now_blob.make_public()

    # Mendapatkan URL gambar yang diunggah secara publik
    img_now_public_url = img_now_blob.public_url

    # Mendapatkan waktu deteksi saat ini
    waktu_deteksi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simpan data counts ke basis data Firestore
    if 'username' in session:
        username = session['username']
        user_ref = db.collection('users').where('username', '==', username).limit(1).get()
        if user_ref:
            user_doc = user_ref[0].to_dict()
            riwayat_pendeteksian = user_doc.get('riwayat_pendeteksian', [])
            riwayat_pendeteksian.append({
                'hasil_deteksi': counts,
                'gambar_sebelum': img_normal_public_url,
                'gambar_setelah': img_now_public_url,
                'waktu_deteksi': waktu_deteksi
            })
            db.collection('users').document(user_ref[0].id).update({'riwayat_pendeteksian': riwayat_pendeteksian})
    
    session['detection_done'] = True

    return render_template("dashboard.html", file_path="img/img_now.jpg", counts=counts, title=title)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    
@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static")

    if not os.path.exists(os.path.join(target, "img")):
        os.makedirs(os.path.join(target, "img"))

    for file in request.files.getlist("file"):
        if file:
            filename = secure_filename(file.filename)
            if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
                file_path = os.path.join(target, "img", "img_now.jpg")
                file.save(file_path)

                copyfile(file_path, os.path.join(target, "img", "img_normal.jpg"))
                session['detection_done'] = False
                session['upload_done'] = True
                return render_template("dashboard.html", file_path=file_path)

    return "Invalid file format. Allowed formats: mp4, png, jpg, jpeg, gif"

@app.route("/dashboard")
def dashboard():
    session['upload_done'] = False
    return render_template("dashboard.html")

@app.route('/informasi-padi')
def informasi_padi():
    with open('info_padi.json', 'r') as f:
        info_padi = json.load(f)
    return render_template('informasi_padi.html', info_padi=info_padi)

@app.route("/logout")
def logout():
    session.clear()
    return render_template("login.html")

@app.route("/riwayat-deteksi")
def riwayat_deteksi():
    if 'username' in session:
        username = session['username']
        user_ref = db.collection('users').where('username', '==', username).limit(1).get()
        if user_ref:
            user_data = user_ref[0].to_dict()
            riwayat_pendeteksian = user_data.get('riwayat_pendeteksian', [])
            return render_template("riwayat_deteksi.html", riwayat_pendeteksian=riwayat_pendeteksian)
    return render_template("riwayat_deteksi.html")

@app.route("/webcam", methods=['GET','POST'])
def webcam():
    return render_template('realtime_detect.html')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

