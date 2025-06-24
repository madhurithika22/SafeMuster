import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import run_pipeline


# app.py
from flask import Flask,jsonify, render_template, Response, request, redirect, url_for
from main_pipeline import run_pipeline
from alert import reverse_geocode, get_nearest_police_station



app = Flask(__name__)




# Global state
location = {"lat": None, "lon": None, "address": "Unknown", "police": "Unknown"}
video_source = 0
current_risk = "Unknown"
current_alert_status = "NO"

def gen_frames():
    global current_risk, current_alert_status
    for frame, risk, alert in run_pipeline(video_source):
        current_risk = risk
        current_alert_status = "YES" if alert else "NO"
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html",
                           address=location["address"],
                           police=location["police"],
                           risk_level=current_risk,
                           alert_status=current_alert_status)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload_video", methods=["POST"])
def upload_video():
    global video_source
    if 'video' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))
    path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(path)
    video_source = path
    return redirect(url_for('index'))

@app.route("/use_webcam", methods=["POST"])
def use_webcam():
    global video_source
    video_source = 0
    return redirect(url_for('index'))

@app.route("/location_update")
def location_update():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
        city, address = reverse_geocode(lat, lon)
        police = get_nearest_police_station(lat, lon)
        location.update({"lat": lat, "lon": lon, "address": address, "police": police})
        return "OK", 200
    except Exception as e:
        print("Location error:", e)
        return "Failed", 400

if __name__ == '__main__':
    app.run(debug=True)
