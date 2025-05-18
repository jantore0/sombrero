import os
import uuid
import shutil
import tempfile
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, jsonify, send_file
import moviepy.editor as mp
import librosa
import numpy as np
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import threading
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Limit uploads to 500MB

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Dictionary to store processing status
processing_status = {}

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Create job directory
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_dir, exist_ok=True)
        os.makedirs(os.path.join(job_dir, "highlight_clips"), exist_ok=True)
        
        # Save the uploaded file
        video_path = os.path.join(job_dir, "videofile.mp4")
        file.save(video_path)
        
        # Initialize status
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting video processing',
            'video_duration': 0,
            'highlight_count': 0,
            'highlight_duration': 0,
            'energy_data': []
        }
        
        # Start processing in background
        thread = threading.Thread(target=process_video, args=(job_id, video_path))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'processing'
        })

def process_video(job_id, video_path):
    """Process video to extract highlights"""
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    highlight_clips_dir = os.path.join(job_dir, "highlight_clips")
    audio_path = os.path.join(job_dir, "audio.wav")
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.mp4")
    
    try:
        # Update status
        update_status(job_id, 10, 'Extracting audio from video')
        
        # 1. Extract audio
        clip = extract_audio(video_path, audio_path)
        video_duration = clip.duration
        update_status(job_id, 20, 'Audio extracted', video_duration=video_duration)
        
        # Update status
        update_status(job_id, 30, 'Analyzing audio for highlights')
        
        # 2. Create highlight clips
        highlight_count, highlight_duration, energy_data = create_highlight_clips(
            audio_path, video_path, highlight_clips_dir, video_duration
        )
        
        update_status(job_id, 60, 'Highlight clips created', 
                     highlight_count=highlight_count,
                     highlight_duration=highlight_duration,
                     energy_data=energy_data)
        
        # Update status
        update_status(job_id, 70, 'Combining highlight clips')
        
        # 3. Combine highlights
        combine_highlights(highlight_clips_dir, result_path)
        
        # Update status
        update_status(job_id, 100, 'Processing completed', 
                     status='completed',
                     result_url=url_for('get_result', job_id=job_id))
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        update_status(job_id, error=str(e))

def update_status(job_id, progress=None, message=None, status=None, 
                 video_duration=None, highlight_count=None, 
                 highlight_duration=None, energy_data=None, error=None):
    """Update processing status for a job"""
    if job_id not in processing_status:
        return
    
    if progress is not None:
        processing_status[job_id]['progress'] = progress
    if message is not None:
        processing_status[job_id]['message'] = message
    if status is not None:
        processing_status[job_id]['status'] = status
    if video_duration is not None:
        processing_status[job_id]['video_duration'] = video_duration
    if highlight_count is not None:
        processing_status[job_id]['highlight_count'] = highlight_count
    if highlight_duration is not None:
        processing_status[job_id]['highlight_duration'] = highlight_duration
    if energy_data is not None:
        processing_status[job_id]['energy_data'] = energy_data
    if error is not None:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {error}'

def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found!")

        clip = mp.VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        
        if not os.path.exists(audio_path):
            raise Exception("Failed to create audio file")
            
        logger.info(f"Audio successfully extracted to {audio_path}")
        logger.info(f"Duration: {clip.duration:.2f} seconds")
        return clip
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def create_highlight_clips(audio_path, video_path, highlight_clips_dir, video_duration):
    """Analyze audio and create highlight clips"""
    try:
        # Load audio
        x, sr = librosa.load(audio_path, sr=16000)
        
        # Get duration
        duration = librosa.get_duration(y=x, sr=sr)
        logger.info(f"Audio loaded. Duration: {duration:.2f} sec ({duration/60:.2f} min)")
        
        # Energy analysis
        max_slice = 2  # Length of segment in seconds (smaller for more granular analysis)
        window_length = max_slice * sr
        
        # Calculate energy for each segment
        energy = np.array([
            sum(abs(x[i:i+window_length]**2)) 
            for i in range(0, len(x), window_length)
        ])
        
        # Normalize energy for visualization
        if len(energy) > 0:
            max_energy = max(energy)
            if max_energy > 0:
                energy = energy / max_energy * 100
        
        # Prepare energy data for frontend
        energy_data = []
        for i, e in enumerate(energy):
            energy_data.append({
                'time': i * max_slice,
                'energy': float(e)
            })
        
        # Set threshold (can be adjusted)
        thresh = np.percentile(energy, 85)  # Automatic threshold (top 15% by energy)
        logger.info(f"Automatic energy threshold: {thresh:.2f}")
        
        # Find interesting moments
        highlights = []
        for i, e in enumerate(energy):
            if e >= thresh:
                start = i * max_slice
                end = (i + 1) * max_slice
                highlights.append((e, start, end))
        
        # Create clips
        highlight_count = 0
        highlight_duration = 0
        for i, (e, start, end) in enumerate(highlights):
            output_file = os.path.join(highlight_clips_dir, f"highlight_{i+1}.mp4")
            logger.info(f"Creating clip {i+1}: {start:.1f}-{end:.1f} sec (energy: {e:.2f})")
            
            try:
                clip_duration = min(duration, end + 2) - max(0, start - 2)
                ffmpeg_extract_subclip(
                    video_path,
                    max(0, start - 2),  # Start 2 sec earlier
                    min(duration, end + 2),  # End 2 sec later
                    targetname=output_file
                )
                highlight_count += 1
                highlight_duration += clip_duration
            except Exception as e:
                logger.error(f"Error creating clip: {str(e)}")
                
        return highlight_count, highlight_duration, energy_data
                
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise

def combine_highlights(highlight_clips_dir, output_path):
    """Combine highlight clips into a single video"""
    try:
        # Check if directory exists
        if not os.path.exists(highlight_clips_dir):
            raise FileNotFoundError(f"Directory {highlight_clips_dir} not found")
        
        # Get list of all highlight files
        clip_files = sorted(glob.glob(os.path.join(highlight_clips_dir, "highlight*.mp4")))
        
        if not clip_files:
            available_files = os.listdir(highlight_clips_dir)
            raise FileNotFoundError(
                f"No highlight files found.\n"
                f"Contents of {highlight_clips_dir}: {available_files}"
            )
        
        # Load clips with error handling
        clips = []
        for file in clip_files:
            try:
                clip = VideoFileClip(file)
                clips.append(clip)
                logger.info(f"Successfully loaded: {file} ({clip.duration:.2f} sec)")
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                continue
        
        if not clips:
            raise ValueError("Could not load any clips")
        
        # Combine clips
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            threads=4
        )
        logger.info(f"\nDone! Created file {output_path} from {len(clips)} clips")
        
    except Exception as e:
        logger.error(f"Error combining highlights: {str(e)}")
        raise

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check the status of a processing job"""
    if job_id in processing_status:
        return jsonify(processing_status[job_id])
    else:
        return jsonify({'status': 'not_found'}), 404

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Serve the result video file"""
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.mp4")
        if not os.path.exists(result_path):
            return jsonify({'error': 'Result not found'}), 404
        return send_file(result_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/templates/<path:filename>')
def serve_static(filename):
    """Serve static files from templates directory"""
    return send_from_directory('templates', filename)

@app.route('/clear/<job_id>', methods=['DELETE'])
def clear_job(job_id):
    """Clean up job files"""
    try:
        # Remove job directory
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        
        # Remove result file
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}.mp4")
        if os.path.exists(result_file):
            os.remove(result_file)
        
        # Remove from status dictionary
        if job_id in processing_status:
            del processing_status[job_id]
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
  
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  