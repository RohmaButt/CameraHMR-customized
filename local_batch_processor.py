# Import dependencies
import cv2
import numpy as np
import os
import subprocess
import time
from pathlib import Path
import glob
import signal
import threading
import queue
import argparse

class BatchProcessor:
    def __init__(self, input_folder, logo_path, output_folder, batch_size=12):
        self.input_folder = input_folder
        self.logo_path = logo_path
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.current_process = None
        self.batch_count = 0
        self.total_frames_processed = 0
        self.status_queue = queue.Queue()

    def cleanup_folders(self):
        """Clean up input and output folders"""
        for folder in [self.input_folder, self.output_folder]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    file_path = os.path.join(folder, f)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")

    def kill_existing_processes(self):
        """Kill any existing processing scripts"""
        try:
            subprocess.run(['pkill', '-f', 'texture_transfer_metric_depth_transparent_optimized.py'],
                         capture_output=True, text=True)
            time.sleep(2)
        except:
            pass

    def start_processing_script(self, use_conda=True, conda_env='camerahmr', use_xvfb=False):
        """Start a new processing script instance"""
        self.kill_existing_processes()

        # Build command based on environment
        if use_xvfb:
            xvfb_prefix = 'xvfb-run -s "-screen 0 640x480x24" '
        else:
            xvfb_prefix = ''
        
        if use_conda:
            python_cmd = f'conda run -n {conda_env} python3'
        else:
            python_cmd = 'python3'  # Use system python or activated environment

        command = (
            f'{xvfb_prefix}{python_cmd} '
            f'texture_transfer_metric_depth_transparent_optimized.py '
            f'--image_folder {self.input_folder} '
            f'--logo_path {self.logo_path} '
            f'--output_folder {self.output_folder} '
            f'--crop_bottom_percentage 0.0 '
            f'--crop_top_percentage 0.0 '
            f'--logo_size 0.8'
        )

        print(f"\n=== Starting Batch {self.batch_count + 1} ===")
        print(f"Command: {command}")

        # Start the process
        self.current_process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name != 'nt' else None  # Unix only
        )

        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_process_output)
        monitor_thread.daemon = True
        monitor_thread.start()

        return self.current_process

    def monitor_process_output(self):
        """Monitor the background process output"""
        if not self.current_process:
            return

        current_frame = None
        processing_stage = ""

        while True:
            output = self.current_process.stdout.readline()
            if output == '' and self.current_process.poll() is not None:
                break
            if output:
                line = output.strip()

                if "Processing frame" in line or "frame_" in line:
                    import re
                    frame_match = re.search(r'frame_(\d+)', line)
                    if frame_match:
                        current_frame = frame_match.group(1)
                        print(f"🎬 [BATCH-{self.batch_count + 1}] Frame {current_frame}: Starting processing...")

                elif "Loading" in line or "Initializing" in line:
                    print(f"⚙️  [BATCH-{self.batch_count + 1}] {line}")

                elif "depth" in line.lower() or "mesh" in line.lower():
                    if current_frame:
                        print(f"🔍 [BATCH-{self.batch_count + 1}] Frame {current_frame}: Generating 3D data")

                elif "texture" in line.lower() or "overlay" in line.lower():
                    if current_frame:
                        print(f"🎨 [BATCH-{self.batch_count + 1}] Frame {current_frame}: Applying texture/overlay")

                elif "Saved" in line or "saved" in line:
                    if current_frame:
                        print(f"✅ [BATCH-{self.batch_count + 1}] Frame {current_frame}: Processing complete!")

                elif "Error" in line or "error" in line or "ERROR" in line:
                    print(f"❌ [BATCH-{self.batch_count + 1}] ERROR: {line}")

            time.sleep(0.01)

    def terminate_current_process(self):
        """Terminate the current processing script"""
        if self.current_process and self.current_process.poll() is None:
            try:
                if os.name != 'nt':  # Unix
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                    time.sleep(2)
                    if self.current_process.poll() is None:
                        os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                else:  # Windows
                    self.current_process.terminate()
                    time.sleep(2)
                    if self.current_process.poll() is None:
                        self.current_process.kill()
            except Exception as e:
                print(f"Error terminating process: {e}")
        self.current_process = None

    def final_file_check(self, expected_files):
        """Final check for files after process completion"""
        final_check = []
        for expected_file in expected_files:
            patterns = [
                expected_file,
                os.path.join(self.output_folder, "*.jpg"),
                os.path.join(self.output_folder, "*.png")
            ]
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    for match in sorted(matches, key=os.path.getmtime, reverse=True):
                        if 'clean_overlay' in match:
                            if os.path.getsize(match) > 0:
                                final_check.append(match)
                                print(f"📁 Found final output: {os.path.basename(match)}")
                                break
                    break
        return final_check

    def wait_for_output(self, expected_files, timeout=120):
        """Wait for expected output files with timeout"""
        timeout_start = time.time()
        completed_frames = set()

        print(f"🔄 Waiting for {len(expected_files)} frames to complete...")

        while time.time() - timeout_start < timeout:
            found_files = []
            newly_completed = []

            for i, expected_file in enumerate(expected_files):
                frame_id = f"frame_{i:04d}"

                if frame_id in completed_frames:
                    found_files.append(expected_file)
                    continue

                if os.path.exists(expected_file):
                    if os.path.getsize(expected_file) > 0:
                        found_files.append(expected_file)
                        if frame_id not in completed_frames:
                            newly_completed.append((frame_id, expected_file))
                            completed_frames.add(frame_id)
                else:
                    base_name = os.path.basename(expected_file)
                    frame_part = base_name.split('_')[0] + '_' + base_name.split('_')[1]
                    patterns = [
                        os.path.join(self.output_folder, f"{frame_part}_*_clean_overlay.jpg"),
                    ]
                    for pattern in patterns:
                        matches = glob.glob(pattern)
                        if matches:
                            match_file = matches[0]
                            if os.path.getsize(match_file) > 0:
                                found_files.append(match_file)
                                if frame_id not in completed_frames:
                                    newly_completed.append((frame_id, match_file))
                                    completed_frames.add(frame_id)
                                break

            for frame_id, output_file in newly_completed:
                print(f"🎉 Frame {frame_id} completed! Output: {os.path.basename(output_file)}")

            if len(found_files) == len(expected_files):
                print(f"✨ All {len(expected_files)} frames in batch completed!")
                return found_files

            if self.current_process and self.current_process.poll() is not None:
                print(f"🏁 Process finished with exit code: {self.current_process.poll()}")
                time.sleep(1)
                final_files = self.final_file_check(expected_files)
                return final_files

            time.sleep(0.2)

        print(f"⏰ Timeout waiting for batch output after {timeout} seconds")
        return found_files if found_files else []


def main():
    parser = argparse.ArgumentParser(description='Webcam batch processor for local machine')
    parser.add_argument('--batch-size', type=int, default=12, help='Number of frames per batch')
    parser.add_argument('--input-folder', type=str, default='./input_frames', help='Input folder path')
    parser.add_argument('--output-folder', type=str, default='./output_frames', help='Output folder path')
    parser.add_argument('--logo-path', type=str, required=True, help='Path to logo image')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera device index (0, 1, 2...)')
    parser.add_argument('--no-conda', action='store_true', help='Do not use conda environment')
    parser.add_argument('--conda-env', type=str, default='camerahmr', help='Conda environment name')
    parser.add_argument('--use-xvfb', action='store_true', help='Use Xvfb for headless display (Linux only)')
    parser.add_argument('--no-display', action='store_true', help='Disable live display window')
    
    args = parser.parse_args()

    # Initialize batch processor
    processor = BatchProcessor(
        args.input_folder,
        args.logo_path,
        args.output_folder,
        args.batch_size
    )

    # Create directories
    os.makedirs(args.input_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # Clean up
    processor.cleanup_folders()
    processor.kill_existing_processes()

    # Initialize webcam
    print(f"Opening camera {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened successfully!")
    print(f"Batch size: {args.batch_size} frames")
    print("Press 'q' to quit, 'c' to capture frame for current batch")

    try:
        frame_count = 0
        batch_frames = []
        batch_frame_paths = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera")
                break

            # Display frame if not disabled
            if not args.no_display:
                display_frame = frame.copy()
                status_text = f"Batch {processor.batch_count + 1} - Frame {len(batch_frames)}/{args.batch_size}"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Webcam Capture', display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture frame
                batch_frames.append(frame.copy())
                
                frame_filename = f'frame_{frame_count:04d}.jpg'
                input_path = os.path.join(args.input_folder, frame_filename)
                cv2.imwrite(input_path, frame)
                batch_frame_paths.append(input_path)
                
                print(f"Captured frame {frame_count} for batch {processor.batch_count + 1}")
                frame_count += 1

            # Auto-process when batch is full
            if len(batch_frames) >= args.batch_size:
                print(f"\n=== Processing Batch {processor.batch_count + 1} ({len(batch_frames)} frames) ===")
                
                process = processor.start_processing_script(
                    use_conda=not args.no_conda,
                    conda_env=args.conda_env,
                    use_xvfb=args.use_xvfb
                )
                
                time.sleep(5)

                if process.poll() is not None:
                    print(f"ERROR: Processing script failed to start")
                    batch_frames.clear()
                    batch_frame_paths.clear()
                    continue

                # Wait for processing
                expected_outputs = []
                for frame_path in batch_frame_paths:
                    base_name = os.path.basename(frame_path).replace('.jpg', '')
                    expected_output = os.path.join(args.output_folder, 
                                                   f'{base_name}_000000_clean_overlay.jpg')
                    expected_outputs.append(expected_output)

                timeout = 300 if processor.batch_count == 0 else 180
                output_files = processor.wait_for_output(expected_outputs, timeout)

                if output_files:
                    print(f"Batch {processor.batch_count + 1} completed successfully!")
                    processor.total_frames_processed += len(batch_frames)
                    
                    # Display results if not disabled
                    if not args.no_display:
                        for output_file in output_files:
                            if os.path.exists(output_file):
                                result_img = cv2.imread(output_file)
                                if result_img is not None:
                                    cv2.imshow('Processed Result', result_img)
                                    cv2.waitKey(500)  # Show for 500ms
                else:
                    print(f"Batch {processor.batch_count + 1} failed or timed out")

                processor.terminate_current_process()
                batch_frames.clear()
                batch_frame_paths.clear()
                processor.batch_count += 1
                
                # Clean up output folder
                try:
                    for f in os.listdir(args.output_folder):
                        if f.endswith(('.png', '.jpg', '.obj', '.mtl')):
                            os.remove(os.path.join(args.output_folder, f))
                except:
                    pass

    except KeyboardInterrupt:
        print("\nStopping due to keyboard interrupt...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.terminate_current_process()
        processor.kill_existing_processes()
        print(f"Pipeline stopped. Total frames processed: {processor.total_frames_processed}")


if __name__ == "__main__":
    main()
