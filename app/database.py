import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Optional

from config import AppConfig


class DatabaseError(RuntimeError):
	pass


class Database:
	def __init__(self, config: AppConfig) -> None:
		self.config = config
		self.driver = config.db_driver

		if self.driver not in {"mysql", "sqlite"}:
			raise DatabaseError("DB_DRIVER harus 'mysql' atau 'sqlite'.")

		self._mysql_connector = None
		if self.driver == "mysql":
			try:
				import mysql.connector  # type: ignore

				self._mysql_connector = mysql.connector
			except ImportError as exc:
				raise DatabaseError(
					"mysql-connector-python belum terpasang. Install dependency terlebih dulu."
				) from exc

	@contextmanager
	def connection(self) -> Generator[Any, None, None]:
		if self.driver == "sqlite":
			db_path = str(self.config.project_root / "app_presensiku.sqlite3")
			conn = sqlite3.connect(db_path)
			conn.row_factory = sqlite3.Row
			try:
				yield conn
				conn.commit()
			finally:
				conn.close()
			return

		conn = self._mysql_connector.connect(
			host=self.config.db_host,
			port=self.config.db_port,
			database=self.config.db_name,
			user=self.config.db_user,
			password=self.config.db_password,
		)
		try:
			yield conn
			conn.commit()
		finally:
			conn.close()

	def init_schema_if_needed(self) -> None:
		attendance = self.config.attendance_table
		if self.driver == "sqlite":
			query = f"""
				CREATE TABLE IF NOT EXISTS {attendance} (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					student_id INTEGER NULL,
					recognized_name TEXT NOT NULL,
					confidence REAL NOT NULL,
					source TEXT NOT NULL,
					attendance_time TEXT NOT NULL,
					created_at TEXT NOT NULL
				)
			"""
			with self.connection() as conn:
				conn.execute(query)
			return

		query = f"""
			CREATE TABLE IF NOT EXISTS {attendance} (
				id BIGINT PRIMARY KEY AUTO_INCREMENT,
				student_id BIGINT NULL,
				recognized_name VARCHAR(255) NOT NULL,
				confidence DOUBLE NOT NULL,
				source VARCHAR(100) NOT NULL,
				attendance_time DATETIME NOT NULL,
				created_at DATETIME NOT NULL
			) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
		"""
		with self.connection() as conn:
			cur = conn.cursor()
			cur.execute(query)
			cur.close()

	def find_student_by_name(self, recognized_name: str) -> Optional[Dict[str, Any]]:
		student_table = self.config.student_table
		id_col = self.config.student_id_column
		name_col = self.config.student_name_column

		if self.driver == "sqlite":
			with self.connection() as conn:
				row = conn.execute(
					f"SELECT {id_col}, {name_col} FROM {student_table} WHERE lower({name_col}) = lower(?) LIMIT 1",
					(recognized_name,),
				).fetchone()
				if row is None:
					return None
				return {"id": row[id_col], "name": row[name_col]}

		with self.connection() as conn:
			cur = conn.cursor(dictionary=True)
			cur.execute(
				f"SELECT {id_col} AS id, {name_col} AS name FROM {student_table} "
				f"WHERE LOWER({name_col}) = LOWER(%s) LIMIT 1",
				(recognized_name,),
			)
			row = cur.fetchone()
			cur.close()
			return row

	def has_attendance_today(self, student_id: Optional[int], recognized_name: str) -> bool:
		attendance_table = self.config.attendance_table
		now = datetime.now()
		day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

		if self.driver == "sqlite":
			with self.connection() as conn:
				if student_id is not None:
					row = conn.execute(
						f"SELECT id FROM {attendance_table} WHERE student_id = ? "
						f"AND attendance_time >= ? LIMIT 1",
						(student_id, day_start.isoformat()),
					).fetchone()
				else:
					row = conn.execute(
						f"SELECT id FROM {attendance_table} WHERE lower(recognized_name) = lower(?) "
						f"AND attendance_time >= ? LIMIT 1",
						(recognized_name, day_start.isoformat()),
					).fetchone()
				return row is not None

		with self.connection() as conn:
			cur = conn.cursor()
			if student_id is not None:
				cur.execute(
					f"SELECT id FROM {attendance_table} WHERE student_id = %s "
					f"AND attendance_time >= %s LIMIT 1",
					(student_id, day_start),
				)
			else:
				cur.execute(
					f"SELECT id FROM {attendance_table} WHERE LOWER(recognized_name) = LOWER(%s) "
					f"AND attendance_time >= %s LIMIT 1",
					(recognized_name, day_start),
				)
			row = cur.fetchone()
			cur.close()
			return row is not None

	def insert_attendance(
		self,
		student_id: Optional[int],
		recognized_name: str,
		confidence: float,
		source: str,
		attendance_time: datetime,
	) -> int:
		attendance_table = self.config.attendance_table

		if self.driver == "sqlite":
			with self.connection() as conn:
				cur = conn.execute(
					f"INSERT INTO {attendance_table} "
					"(student_id, recognized_name, confidence, source, attendance_time, created_at) "
					"VALUES (?, ?, ?, ?, ?, ?)",
					(
						student_id,
						recognized_name,
						confidence,
						source,
						attendance_time.isoformat(),
						datetime.now().isoformat(),
					),
				)
				return int(cur.lastrowid)

		with self.connection() as conn:
			cur = conn.cursor()
			cur.execute(
				f"INSERT INTO {attendance_table} "
				"(student_id, recognized_name, confidence, source, attendance_time, created_at) "
				"VALUES (%s, %s, %s, %s, %s, %s)",
				(
					student_id,
					recognized_name,
					confidence,
					source,
					attendance_time,
					datetime.now(),
				),
			)
			attendance_id = int(cur.lastrowid)
			cur.close()
			return attendance_id
