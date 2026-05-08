"""Flask app that exposes health, training, and attendance endpoints."""

import cv2
import numpy as np
from flask import Flask, jsonify, request
from pathlib import Path
from datetime import datetime
import requests
from io import BytesIO

from config import AppConfig
from database import Database, DatabaseError
from services.attendance import AttendanceService
from services.camera_absensi import CameraAttendanceRunner
from services.camera_gui import CameraAttendanceGUI
from services.inference import FacePredictor
from services.preprocess import preprocess_dataset
from services.train_cnn import train_model
from services.fetch_laravel_dataset import LaravelDatasetFetcher
from services.attendance_utils import calculate_attendance_status, format_attendance_status


def _decode_uploaded_image() -> np.ndarray:
	"""Read the uploaded file and convert it into an OpenCV image."""
	if "image" not in request.files:
		raise ValueError("Field file 'image' wajib diisi.")

	file = request.files["image"]
	image_bytes = file.read()
	img_array = np.frombuffer(image_bytes, np.uint8)
	frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	if frame is None:
		raise ValueError("Gagal decode file gambar.")
	return frame


def _save_attendance_picture(frame: np.ndarray, config: AppConfig) -> str:
	"""Save attendance picture to Laravel storage and return filename.
	
	Args:
	- frame: OpenCV image (BGR)
	- config: App configuration
	
	Returns: Relative path for database storage 
	         (e.g., 'daily_attendance_pictures/2026-02-05_14-30-45_123.jpg')
	"""
	try:
		# Generate filename
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
		filename = f"{timestamp}.jpg"
		
		# Encode frame to JPG
		success, jpg_buffer = cv2.imencode(".jpg", frame)
		if not success:
			raise ValueError("Gagal encode gambar ke JPG")
		
		jpg_bytes = jpg_buffer.tobytes()
		
		# Upload to Laravel via API
		url = f"{config.laravel_url}/api/attendance/upload-picture"
		
		files = {"image": (filename, BytesIO(jpg_bytes), "image/jpeg")}
		data = {"storage_path": config.laravel_attendance_pictures_path}
		
		print(f"[UPLOAD] Sending to: {url}")
		print(f"[UPLOAD] Storage path: {config.laravel_attendance_pictures_path}")
		print(f"[UPLOAD] Filename: {filename}")
		
		response = requests.post(url, files=files, data=data, timeout=10)
		
		print(f"[UPLOAD] Response status: {response.status_code}")
		print(f"[UPLOAD] Response headers: {response.headers}")
		print(f"[UPLOAD] Response body (first 500 chars): {response.text[:500]}")
		
		if response.status_code == 200:
			try:
				result = response.json()
				print(f"[UPLOAD] Response JSON: {result}")
				uploaded_path = result.get("path", f"{config.laravel_attendance_pictures_path}/{filename}")
				print(f"[UPLOAD] Success! Path: {uploaded_path}")
				return uploaded_path
			except Exception as json_error:
				print(f"[UPLOAD] JSON parse error: {str(json_error)}")
				print(f"[UPLOAD] Fallback to local storage")
				return _save_attendance_picture_locally(frame, config, filename)
		else:
			# Log error response
			print(f"[UPLOAD] Error response: {response.text}")
			print(f"[UPLOAD] Fallback to local storage")
			return _save_attendance_picture_locally(frame, config, filename)
			
	except requests.exceptions.ConnectionError as e:
		print(f"[UPLOAD] Connection error to Laravel: {str(e)}")
		print(f"[UPLOAD] Fallback to local storage")
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
		filename = f"{timestamp}.jpg"
		return _save_attendance_picture_locally(frame, config, filename)
	except Exception as e:
		print(f"[UPLOAD] Unexpected error: {str(e)}")
		print(f"[UPLOAD] Fallback to local storage")
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
		filename = f"{timestamp}.jpg"
		return _save_attendance_picture_locally(frame, config, filename)


def _save_attendance_picture_locally(
	frame: np.ndarray, config: AppConfig, filename: str = None
) -> str:
	"""Fallback: Save attendance picture to local disk.
	
	Args:
	- frame: OpenCV image (BGR)
	- config: App configuration
	- filename: Optional filename (default: timestamp)
	
	Returns: Relative path (e.g., 'attendance_pictures/2026-02-05_14-30-45_123.jpg')
	"""
	if filename is None:
		timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
		filename = f"{timestamp}.jpg"
	
	picture_dir = config.project_root / "attendance_pictures"
	picture_dir.mkdir(exist_ok=True)
	
	filepath = picture_dir / filename
	cv2.imwrite(str(filepath), frame)
	
	# Return relative path for database storage
	return f"attendance_pictures/{filename}"


def _read_pipeline_options(config: AppConfig) -> dict:
	"""Collect pipeline options from the request body."""
	payload = request.get_json(silent=True) or {}

	return {
		"fetch_from_laravel": bool(payload.get("fetch_from_laravel", False)),
		"run_preprocess": bool(payload.get("preprocess", True)),
		"run_train": bool(payload.get("train", True)),
		"overwrite": bool(payload.get("overwrite", False)),
		"epochs": int(payload.get("epochs", config.epochs)),
		"min_images": int(payload.get("min_images_per_class", config.min_images_per_class)),
		"train_ratio": float(payload.get("train_ratio", config.train_ratio)),
		"val_ratio": float(payload.get("val_ratio", config.val_ratio)),
		"test_ratio": float(payload.get("test_ratio", config.test_ratio)),
	}


def create_app() -> Flask:
	app = Flask(__name__)
	config = AppConfig.from_env()

	db = Database(config)
	# Note: Database schema already exists in Laravel database
	# No need to call db.init_schema_if_needed()

	predictor = FacePredictor(
		model_path=config.model_path,
		class_names_path=config.class_names_path,
		image_size=config.image_size,
	)
	attendance_service = AttendanceService(db)
	camera_runner = CameraAttendanceRunner(config, predictor, attendance_service)
	camera_gui_runner = CameraAttendanceGUI(config, predictor, attendance_service)

	@app.get("/health")
	def health():
		status = {
			"status": "ok",
			"model_exists": config.model_path.exists(),
			"dataset_root": str(config.dataset_root),
		}
		return jsonify(status)

	@app.post("/pipeline/run")
	def run_pipeline():
		options = _read_pipeline_options(config)

		results = {}
		
		# Fetch dataset from Laravel if requested
		if options["fetch_from_laravel"]:
			try:
				fetcher = LaravelDatasetFetcher(
					db=db,
					laravel_url=config.laravel_url,
					storage_path=config.laravel_storage_path,
				)
				fetch_result = fetcher.fetch_and_organize(
					output_dir=config.raw_dir,
					overwrite=options["overwrite"],
				)
				results["fetch_laravel"] = fetch_result
				
				if not fetch_result["success"]:
					return jsonify({
						"status": "error",
						"message": "Gagal fetch dataset dari Laravel",
						"details": fetch_result,
					}), 400
				
				# Cleanup and reorganize
				cleanup_result = fetcher.cleanup_and_reorganize(
					dataset_dir=config.raw_dir,
					target_size=(config.image_size, config.image_size),
				)
				results["cleanup"] = cleanup_result
				
			except Exception as e:
				return jsonify({
					"status": "error",
					"message": f"Error fetch dari Laravel: {str(e)}",
				}), 500
		
		if options["run_preprocess"]:
			results["preprocess"] = preprocess_dataset(
				source_dir=config.raw_dir,
				output_dir=config.preprocessed_dir,
				target_size=config.image_size,
				min_images_per_class=options["min_images"],
				seed=config.random_seed,
				overwrite=options["overwrite"],
			)

		if options["run_train"]:
			results["train"] = train_model(
				dataset_dir=config.preprocessed_dir,
				model_path=config.model_path,
				class_names_path=config.class_names_path,
				logs_dir=config.logs_dir,
				image_size=config.image_size,
				batch_size=config.batch_size,
				epochs=options["epochs"],
				learning_rate=config.learning_rate,
				seed=config.random_seed,
				train_ratio=options["train_ratio"],
				val_ratio=options["val_ratio"],
				test_ratio=options["test_ratio"],
			)

		return jsonify({"message": "pipeline selesai", "results": results})

	@app.post("/attendance/recognize")
	def recognize_and_attend():
		frame = _decode_uploaded_image()
		prediction = predictor.predict(frame)

		if prediction is None:
			return jsonify({"status": "no_face", "message": "Wajah tidak terdeteksi."}), 200

		if prediction.confidence < config.threshold:
			unknown_response = {
				"status": "unknown",
				"name": prediction.recognized_name,
				"confidence": prediction.confidence,
				"message": "Prediksi di bawah threshold.",
			}
			return jsonify(unknown_response), 200

		# Calculate attendance status (tepat_waktu or terlambat)
		# Batas masuk berdasarkan config (default: jam 7:00)
		attendance_status = calculate_attendance_status(
			cutoff_hour=config.attendance_cutoff_hour,
			cutoff_minute=config.attendance_cutoff_minute,
		)

		# Save the picture
		picture_filename = _save_attendance_picture(frame, config)

		attendance = attendance_service.mark_attendance(
			recognized_name=prediction.recognized_name,
			confidence=prediction.confidence,
			source="upload",
			picture_filename=picture_filename,
			status=attendance_status,
		)

		return jsonify(
			{
				"status": attendance.status,
				"message": attendance.message,
				"name": prediction.recognized_name,
				"confidence": prediction.confidence,
				"student_name": attendance.student_name,
				"attendance_id": attendance.attendance_id,
				"student_id": attendance.student_id,
				"class_id": attendance.class_id,
				"attendance_status": attendance_status,
				"attendance_status_display": format_attendance_status(attendance_status),
				"picture": picture_filename,
			}
		)

	@app.post("/attendance/camera/start")
	def start_camera_attendance():
		"""Start camera attendance.
		
		Query Parameters:
		- gui (bool, optional): Enable GUI display (default: false)
		
		Examples:
		  POST /attendance/camera/start  (no GUI)
		  POST /attendance/camera/start?gui=true  (with GUI)
		  POST /attendance/camera/start?gui=1  (with GUI)
		"""
		gui_enabled = request.args.get('gui', 'false').lower() in {'true', '1', 'yes'}
		
		if gui_enabled:
			try:
				camera_gui_runner.start()
				return jsonify({
					"status": "started",
					"mode": "gui",
					"message": "Camera attendance dimulai dengan GUI"
				})
			except Exception as e:
				return jsonify({
					"status": "error",
					"message": f"Error starting camera GUI: {str(e)}"
				}), 500
		else:
			camera_runner.start()
			return jsonify({
				"status": "started",
				"mode": "background",
				"message": "Camera attendance dimulai (background mode)"
			})

	@app.post("/attendance/camera/stop")
	def stop_camera_attendance():
		"""Stop camera attendance (both GUI and background mode)."""
		camera_runner.stop()
		camera_gui_runner.stop()
		return jsonify({"status": "stopped", "message": "Camera attendance dihentikan"})

	@app.get("/attendance/camera/status")
	def camera_status():
		state = camera_runner.state()
		return jsonify(
			{
				"running": state.running,
				"last_identity": state.last_identity,
				"last_confidence": state.last_confidence,
				"last_message": state.last_message,
			}
		)

	@app.errorhandler(DatabaseError)
	def handle_db_error(err):
		return jsonify({"error": str(err)}), 500

	@app.errorhandler(ValueError)
	def handle_value_error(err):
		return jsonify({"error": str(err)}), 400

	
	@app.post("/dataset/fetch-from-laravel")
	def fetch_dataset_from_laravel():
		"""Fetch student photos from Laravel storage and organize into dataset."""
		try:
			payload = request.get_json(silent=True) or {}
			overwrite = bool(payload.get("overwrite", False))
			
			fetcher = LaravelDatasetFetcher(
				db=db,
				laravel_url=config.laravel_url,
				storage_path=config.laravel_storage_path,
			)
			
			# Fetch and organize
			fetch_result = fetcher.fetch_and_organize(
				output_dir=config.raw_dir,
				overwrite=overwrite,
			)
			
			if not fetch_result["success"]:
				return jsonify({
					"status": "error",
					"message": "Gagal fetch dataset dari Laravel",
					"details": fetch_result,
				}), 400
			
			# Cleanup and reorganize images
			cleanup_result = fetcher.cleanup_and_reorganize(
				dataset_dir=config.raw_dir,
				target_size=(config.image_size, config.image_size),
			)
			
			return jsonify({
				"status": "success",
				"message": "Dataset berhasil di-fetch dari Laravel",
				"fetch_result": fetch_result,
				"cleanup_result": cleanup_result,
			}), 200
			
		except Exception as e:
			return jsonify({
				"status": "error",
				"message": f"Error fetch dataset: {str(e)}",
			}), 500

	@app.errorhandler(FileNotFoundError)
	def handle_not_found(err):
		return jsonify({"error": str(err)}), 404

	@app.errorhandler(Exception)
	def handle_unexpected(err):
		return jsonify({"error": f"Unexpected error: {err}"}), 500

	return app


if __name__ == "__main__":
	flask_app = create_app()
	cfg = AppConfig.from_env()
	flask_app.run(host="0.0.0.0", port=5000, debug=cfg.debug)
