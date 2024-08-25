import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import boto3
import os
import soundfile as sf
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from PIL import Image
# กำหนดค่า AWS S3
bucket_name = 'my-watermelon-models'
model_file_name = '3.h5'
model_file_path = 'model/3.h5'

# กำหนด AWS credentials และ Region จาก Streamlit secrets
aws_access_key_id = 'AKIAQKGGXRGHVXFZREWH'
aws_secret_access_key = 'TcyEltWdw5VyIu0YO5XdfwcRJQLTXt/FCLD9JJKU'
region_name = 'ap-southeast-1'

# Load your watermelon images
unripe_image = Image.open("image/watermelon_unripe.jpg")
semiripe_image = Image.open("image/watermelon_semiripe.jpg")
ripe_image = Image.open("image/watermelon_ripe.jpg")

# ตรวจสอบว่าโฟลเดอร์ model มีอยู่หรือไม่ ถ้าไม่มีให้สร้าง
if not os.path.exists('model'):
    os.makedirs('model')

# ดาวน์โหลดโมเดลจาก S3 พร้อมจัดการข้อผิดพลาด
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

try:
    if not os.path.exists(model_file_path):
        st.info(f"Downloading {model_file_name} from S3 bucket {bucket_name}...")
        s3.download_file(bucket_name, model_file_name, model_file_path)
        st.success("Model downloaded successfully.")
except s3.exceptions.NoSuchBucket:
    st.error(f"The specified bucket does not exist: {bucket_name}")
except s3.exceptions.NoSuchKey:
    st.error(f"The specified key does not exist: {model_file_name}")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# เพิ่มการจัดการข้อผิดพลาดเมื่อโหลดโมเดล
try:
    model = tf.keras.models.load_model(model_file_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()
    
def preprocess_audio_file(file_path, max_pad_len=174):
    try:
        # ใช้ pydub เพื่อเปิดไฟล์เสียงและแปลงเป็น wav
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # ตั้งค่า sample rate และ channels
        temp_wav_path = "temp.wav"
        audio.export(temp_wav_path, format="wav")
        
        data, sample_rate = librosa.load(temp_wav_path)
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        zcr = librosa.feature.zero_crossing_rate(data)
        chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)

        feature_len = max(mfccs.shape[1], zcr.shape[1], chroma.shape[1])
        pad_width = max_pad_len - feature_len
        
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            zcr = np.pad(zcr, pad_width=((0, pad_width)), mode='constant')
            chroma = np.pad(chroma, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            zcr = zcr[:, :max_pad_len]
            chroma = chroma[:, :max_pad_len]

        # รวม MFCCs, ZCR, และ Chroma เข้าด้วยกัน
        combined_feature = np.vstack([mfccs, zcr, chroma])
        combined_feature = np.pad(combined_feature, pad_width=((0, model.input_shape[1] - combined_feature.shape[0]), (0, 0)), mode='constant')
        combined_feature = np.expand_dims(combined_feature, axis=-1)

        return combined_feature
    except FileNotFoundError as e:
        st.error("ffmpeg not found. Please ensure ffmpeg is installed and added to PATH.")
        raise e

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.recording_complete = False

    def recv(self, frame):
        audio_data = np.frombuffer(frame.to_ndarray(), np.float32)
        self.audio_buffer.extend(audio_data)

        if len(self.audio_buffer) > 16000 * 5:  # 5 seconds buffer
            audio_segment = np.array(self.audio_buffer[:16000 * 5])
            self.audio_buffer = []
            self.recording_complete = True

            # บันทึกเสียงเป็นไฟล์ .wav
            sf.write('recorded_audio.wav', audio_segment, 16000)

            st.session_state['recorded_audio'] = 'recorded_audio.wav'
            st.session_state['result'] = "การบันทึกเสียงเสร็จสมบูรณ์ กรุณาอัปโหลดไฟล์เสียงที่บันทึกไว้เพื่อทำการประมวลผล"

        return frame

# สร้างอินสแตนซ์ของ AudioProcessor
audio_processor = AudioProcessor()

st.title('แอพจำแนกความสุกของแตงโม')

webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, audio_processor_factory=lambda: audio_processor, rtc_configuration={
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}, media_stream_constraints={
    "audio": True,
    "video": False,
})

# แสดงข้อความเมื่อการบันทึกเสียงเสร็จสมบูรณ์
if audio_processor.recording_complete and 'result' in st.session_state:
    st.write(st.session_state['result'])
    st.download_button(
        label="Download recorded audio",
        data=open(st.session_state['recorded_audio'], 'rb'),
        file_name="recorded_audio.wav",
        mime="audio/wav"
    )

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียงหรือวิดีโอ", type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format='audio/wav')

    try:
        if file_path.endswith(('.mp4', '.mov', '.avi')):
            video = VideoFileClip(file_path)
            audio = video.audio
            audio.write_audiofile("temp_audio.wav")
            file_path = "temp_audio.wav"

        processed_data = preprocess_audio_file(file_path)
        prediction = model.predict(np.expand_dims(processed_data, axis=0))
        predicted_class = np.argmax(prediction)

        # Map the prediction to display the corresponding image and label
        if predicted_class == 0:
            st.write("แตงโมสุก (แตงโมที่มีเนื้อเป็นสีแดงเข้ม)")
            #result = 'แตงโมสุก (แตงโมที่มีเนื้อเป็นสีแดงเข้ม)'
            st.image(ripe_image)
            
        elif predicted_class == 1:
            st.write("แตงโมกึ่งสุก (แตงโมที่มีเนื้อเป็นสีแดงอ่อน)")
            #result = 'แตงโมกึ่งสุก (แตงโมที่มีเนื้อเป็นสีแดงอ่อน)'
            st.image(semiripe_image)
        else:
            st.write("แตงโมไม่สุก (แตงโมที่มีเนื้อเป็นขาวอมชมพู) หรืออาจไม่ใช่เสียงการเคาะแตงโม")
            #result = 'แตงโมไม่สุก (แตงโมที่มีเนื้อเป็นขาวอมชมพู) หรืออาจไม่ใช่เสียงการเคาะแตงโม'
            st.image(unripe_image)

        # Display confidence score
        #confidence = np.max(prediction)
        #st.write(f"ความมั่นใจของการทำนาย: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
