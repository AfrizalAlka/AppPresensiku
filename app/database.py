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

	def fetch_all(self, query: str, params: tuple = ()) -> list:
		"""Execute a query and fetch all results.
		
		Args:
			query: SQL query string
			params: Query parameters (for parameterized queries)
		
		Returns: List of tuples (for sqlite) or list of dicts (for mysql)
		"""
		if self.driver == "sqlite":
			with self.connection() as conn:
				rows = conn.execute(query, params).fetchall()
				return rows or []
		
		with self.connection() as conn:
			cur = conn.cursor(dictionary=True)
			cur.execute(query, params)
			rows = cur.fetchall()
			cur.close()
			return rows or []

	def find_student_by_name(self, recognized_name: str) -> Optional[Dict[str, Any]]:
		"""Find student by name from Laravel students table.
		
		Returns: {"id": int, "name": str, "id_class": int} or None
		"""
		student_table = self.config.student_table
		id_col = self.config.student_id_column
		name_col = self.config.student_name_column
		class_col = self.config.student_class_column

		if self.driver == "sqlite":
			with self.connection() as conn:
				row = conn.execute(
					f"SELECT {id_col}, {name_col}, {class_col} FROM {student_table} WHERE lower({name_col}) = lower(?) LIMIT 1",
					(recognized_name,),
				).fetchone()
				if row is None:
					return None
				return {
					"id": row[id_col],
					"name": row[name_col],
					"id_class": row[class_col]
				}

		with self.connection() as conn:
			cur = conn.cursor(dictionary=True)
			cur.execute(
				f"SELECT {id_col} AS id, {name_col} AS name, {class_col} AS id_class FROM {student_table} "
				f"WHERE LOWER({name_col}) = LOWER(%s) LIMIT 1",
				(recognized_name,),
			)
			row = cur.fetchone()
			cur.close()
			return row

	def has_attendance_today(self, student_id: int, id_class: int) -> bool:
		"""Check if student already has attendance recorded today for this class."""
		attendance_table = self.config.attendance_table
		now = datetime.now()
		day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

		if self.driver == "sqlite":
			with self.connection() as conn:
				row = conn.execute(
					f"SELECT id FROM {attendance_table} WHERE id_student = ? "
					f"AND id_class = ? AND created_at >= ? LIMIT 1",
					(student_id, id_class, day_start.isoformat()),
				).fetchone()
				return row is not None

		with self.connection() as conn:
			cur = conn.cursor()
			cur.execute(
				f"SELECT id FROM {attendance_table} WHERE id_student = %s "
				f"AND id_class = %s AND DATE(created_at) = DATE(%s) LIMIT 1",
				(student_id, id_class, now),
			)
			row = cur.fetchone()
			cur.close()
			return row is not None

	def insert_attendance(
		self,
		id_student: int,
		id_class: int,
		picture_filename: str,
		status: str = "tepat waktu",
	) -> int:
		"""Insert attendance record into attendance_history_dailys table.
		
		Args:
		- id_student: Student ID from students table
		- id_class: Class ID from clases table
		- picture_filename: Name/path of the picture file
		- status: 'tepat waktu' or 'terlambat'
		
		Returns: Attendance ID
		"""
		attendance_table = self.config.attendance_table
		now = datetime.now()

		if self.driver == "sqlite":
			with self.connection() as conn:
				cur = conn.execute(
					f"INSERT INTO {attendance_table} "
					"(id_student, id_class, picture, status, created_at, updated_at) "
					"VALUES (?, ?, ?, ?, ?, ?)",
					(
						id_student,
						id_class,
						picture_filename,
						status,
						now.isoformat(),
						now.isoformat(),
					),
				)
				return int(cur.lastrowid)

		with self.connection() as conn:
			cur = conn.cursor()
			cur.execute(
				f"INSERT INTO {attendance_table} "
				"(id_student, id_class, picture, status, created_at, updated_at) "
				"VALUES (%s, %s, %s, %s, %s, %s)",
				(
					id_student,
					id_class,
					picture_filename,
					status,
					now,
					now,
				),
			)
			attendance_id = int(cur.lastrowid)
			cur.close()
			return attendance_id

