import cv2
import numpy as np
from flask import Flask, jsonify, request

from config import AppConfig
from database import Database, DatabaseError
from services.attendance import AttendanceService
from services.camera_absensi import CameraAttendanceRunner
from services.inference import FacePredictor
from services.preprocess import preprocess_dataset
from services.train_cnn import train_model


def _decode_uploaded_image() -> np.ndarray:
	if "image" not in request.files:
		raise ValueError("Field file 'image' wajib diisi.")

	file = request.files["image"]
	image_bytes = file.read()
	img_array = np.frombuffer(image_bytes, np.uint8)
	frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
	if frame is None:
		raise ValueError("Gagal decode file gambar.")
	return frame


def create_app() -> Flask:
	app = Flask(__name__)
	config = AppConfig.from_env()

	db = Database(config)
	db.init_schema_if_needed()

	predictor = FacePredictor(
		model_path=config.model_path,
		class_names_path=config.class_names_path,
		image_size=config.image_size,
	)
	attendance_service = AttendanceService(db)
	camera_runner = CameraAttendanceRunner(config, predictor, attendance_service)

	@app.get("/health")
	def health():
		return jsonify(
			{
				"status": "ok",
				"model_exists": config.model_path.exists(),
				"dataset_root": str(config.dataset_root),
			}
		)

	@app.post("/pipeline/run")
	def run_pipeline():
		payload = request.get_json(silent=True) or {}

		run_preprocess = bool(payload.get("preprocess", True))
		run_train = bool(payload.get("train", True))
		overwrite = bool(payload.get("overwrite", False))

		epochs = int(payload.get("epochs", config.epochs))
		min_images = int(payload.get("min_images_per_class", config.min_images_per_class))
		train_ratio = float(payload.get("train_ratio", config.train_ratio))
		val_ratio = float(payload.get("val_ratio", config.val_ratio))
		test_ratio = float(payload.get("test_ratio", config.test_ratio))

		results = {}
		if run_preprocess:
			results["preprocess"] = preprocess_dataset(
				source_dir=config.raw_dir,
				output_dir=config.preprocessed_dir,
				target_size=config.image_size,
				min_images_per_class=min_images,
				seed=config.random_seed,
				overwrite=overwrite,
			)

		if run_train:
			results["train"] = train_model(
				dataset_dir=config.preprocessed_dir,
				model_path=config.model_path,
				class_names_path=config.class_names_path,
				logs_dir=config.logs_dir,
				image_size=config.image_size,
				batch_size=config.batch_size,
				epochs=epochs,
				learning_rate=config.learning_rate,
				seed=config.random_seed,
				train_ratio=train_ratio,
				val_ratio=val_ratio,
				test_ratio=test_ratio,
			)

		return jsonify({"message": "pipeline selesai", "results": results})

	@app.post("/attendance/recognize")
	def recognize_and_attend():
		frame = _decode_uploaded_image()
		prediction = predictor.predict(frame)

		if prediction is None:
			return jsonify({"status": "no_face", "message": "Wajah tidak terdeteksi."}), 200

		if prediction.confidence < config.threshold:
			return jsonify(
				{
					"status": "unknown",
					"name": prediction.recognized_name,
					"confidence": prediction.confidence,
					"message": "Prediksi di bawah threshold.",
				}
			), 200

		attendance = attendance_service.mark_attendance(
			recognized_name=prediction.recognized_name,
			confidence=prediction.confidence,
			source="upload",
		)

		return jsonify(
			{
				"status": attendance.status,
				"message": attendance.message,
				"name": prediction.recognized_name,
				"confidence": prediction.confidence,
				"attendance_id": attendance.attendance_id,
				"student_id": attendance.student_id,
			}
		)

	@app.post("/attendance/camera/start")
	def start_camera_attendance():
		camera_runner.start()
		return jsonify({"status": "started"})

	@app.post("/attendance/camera/stop")
	def stop_camera_attendance():
		camera_runner.stop()
		return jsonify({"status": "stopped"})

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
