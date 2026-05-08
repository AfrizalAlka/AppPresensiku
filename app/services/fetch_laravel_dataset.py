"""Service to fetch student photos from Laravel storage and organize into dataset folders."""

import os
import shutil
from pathlib import Path
from typing import Optional
import requests
import cv2
import numpy as np

from database import Database


class LaravelDatasetFetcher:
    """Fetch student photos from Laravel storage and organize into training dataset."""

    def __init__(self, db: Database, laravel_url: str, storage_path: str):
        """Initialize fetcher.
        
        Args:
            db: Database connection
            laravel_url: Base URL of Laravel app (e.g., 'https://example.com')
            storage_path: Storage path (e.g., 'photo-webcam')
        """
        self.db = db
        self.laravel_url = laravel_url.rstrip("/")
        self.storage_path = storage_path.strip("/")

    def fetch_and_organize(
        self, output_dir: Path, overwrite: bool = False
    ) -> dict:
        """Fetch photos from Laravel and organize into dataset structure.
        
        Args:
            output_dir: Output directory for organized dataset
            overwrite: Whether to overwrite existing files
            
        Returns:
            {
                'success': True/False,
                'students_processed': int,
                'photos_downloaded': int,
                'photos_failed': int,
                'errors': [error messages]
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "success": True,
            "students_processed": 0,
            "photos_downloaded": 0,
            "photos_failed": 0,
            "errors": [],
        }

        try:
            # Query students with pictures
            students = self._get_students_with_pictures()
            results["students_processed"] = len(students)

            for student_id, student_name, pictures_str in students:
                if not pictures_str:
                    continue

                # Create folder for student
                student_folder = output_dir / student_name
                if student_folder.exists() and not overwrite:
                    results["errors"].append(
                        f"Folder '{student_name}' sudah ada, skipped"
                    )
                    continue

                student_folder.mkdir(parents=True, exist_ok=True)

                # Parse picture filenames (comma-separated)
                picture_filenames = [
                    f.strip() for f in pictures_str.split(",") if f.strip()
                ]

                for filename in picture_filenames:
                    try:
                        self._download_and_save_photo(
                            filename, student_folder, student_name
                        )
                        results["photos_downloaded"] += 1
                    except Exception as e:
                        results["photos_failed"] += 1
                        results["errors"].append(
                            f"Student: {student_name}, File: {filename} - {str(e)}"
                        )

        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Fetch failed: {str(e)}")

        return results

    def _get_students_with_pictures(self) -> list:
        """Get students that have pictures from database.
        
        Returns:
            List of tuples: [(id, name, pictures_str), ...]
        """
        query = """
            SELECT id, name, pictures 
            FROM students 
            WHERE pictures IS NOT NULL AND pictures != ''
            ORDER BY name
        """
        rows = self.db.fetch_all(query)
        
        # Convert rows to list of tuples for consistent handling of sqlite and mysql results
        result = []
        for row in rows:
            if isinstance(row, dict):
                # MySQL result (dictionary)
                result.append((row['id'], row['name'], row['pictures']))
            else:
                # SQLite result (tuple/Row object)
                result.append((row[0], row[1], row[2]))
        
        return result

    def _download_and_save_photo(
        self, filename: str, student_folder: Path, student_name: str
    ) -> None:
        """Download photo from Laravel storage and save locally.
        
        Args:
            filename: Filename from database (e.g., 'webcam_1234567_1_abc123.png')
            student_folder: Folder to save the photo
            student_name: Student name for logging
        """
        # Build URL to photo
        url = (
            f"{self.laravel_url}/storage/{self.storage_path}/{filename}"
        )

        # Download with timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Save locally
        output_file = student_folder / filename
        with open(output_file, "wb") as f:
            f.write(response.content)

        # Verify it's a valid image
        img = cv2.imread(str(output_file))
        if img is None:
            output_file.unlink()  # Delete invalid file
            raise ValueError(f"Downloaded file is not a valid image: {filename}")

    def cleanup_and_reorganize(
        self, dataset_dir: Path, target_size: tuple = (224, 224)
    ) -> dict:
        """Cleanup corrupted images and resize to target size.
        
        Args:
            dataset_dir: Dataset directory with student folders
            target_size: Target image size (width, height)
            
        Returns:
            {'cleaned': int, 'resized': int, 'removed': int, 'errors': []}
        """
        dataset_dir = Path(dataset_dir)
        results = {"cleaned": 0, "resized": 0, "removed": 0, "errors": []}

        try:
            for student_folder in dataset_dir.iterdir():
                if not student_folder.is_dir():
                    continue

                for img_file in student_folder.glob("*"):
                    try:
                        # Try to read image
                        img = cv2.imread(str(img_file))
                        if img is None:
                            img_file.unlink()
                            results["removed"] += 1
                            continue

                        # Resize to target size
                        img_resized = cv2.resize(img, target_size)
                        cv2.imwrite(str(img_file), img_resized)
                        results["resized"] += 1
                        results["cleaned"] += 1

                    except Exception as e:
                        results["errors"].append(
                            f"{img_file.name}: {str(e)}"
                        )
                        try:
                            img_file.unlink()
                            results["removed"] += 1
                        except:
                            pass

        except Exception as e:
            results["errors"].append(f"Cleanup failed: {str(e)}")

        return results
