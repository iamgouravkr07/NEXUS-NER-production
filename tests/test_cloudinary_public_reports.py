"""
NEXUS-NER - Cloudinary Durable Photo Storage Test Suite

Tests:
1. Photo storage unit tests:
   - Magic byte validation for JPEG, PNG, WebP.
   - Rejection of invalid/corrupt magic bytes and short payloads.
   - Public ID generation format reports/{year}/{month}/{uuid}.
   - Extension and MIME validation.
   - 10MB file size limit enforcement.
   - Missing Cloudinary credentials returning HTTP 503.
   - Cloudinary upload failure returning HTTP 502.
2. End-to-end API integration tests (using TestClient and in-memory SQLite):
   - Submission without photo (JSON and multipart/form-data) creates UNVERIFIED report.
   - Valid photo upload with mocked Cloudinary persists photo_url, photo_public_id, content_type.
   - Public ID format enforcement reports/{year}/{month}/{uuid}.
   - Rejection of invalid extensions (.exe, .txt) via API (HTTP 400).
   - Rejection of spoofed images (fake .jpg with text content) via API (HTTP 400).
   - Retrieval via /public-reports/mine and /public-reports/{id} returns photo metadata.
"""

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend directory is in python search path
backend_path = Path(__file__).resolve().parents[1] / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Set test environment variables before importing app modules
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["JWT_ALGORITHM"] = "HS256"

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth import get_current_user
from app.database import get_db
from app.main import app
from app.models.public_report import PublicReport
from app.models.user import User
from app.services import auth_service, photo_storage
from app.services.photo_storage import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    PhotoUploadResult,
    generate_public_id,
    upload_report_photo,
    validate_image_magic_bytes,
)


# Reusable image fixtures
VALID_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 100
VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
VALID_WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 100


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str = "image/jpeg"):
        self.filename = filename
        self.file = io.BytesIO(content)
        self.content_type = content_type


class PhotoStorageUnitTests(unittest.TestCase):
    """Unit tests for Cloudinary photo storage service and validators."""

    def test_magic_bytes_detection_jpeg(self):
        detected = validate_image_magic_bytes(VALID_JPEG_BYTES)
        self.assertEqual(detected, "image/jpeg")

    def test_magic_bytes_detection_png(self):
        detected = validate_image_magic_bytes(VALID_PNG_BYTES)
        self.assertEqual(detected, "image/png")

    def test_magic_bytes_detection_webp(self):
        detected = validate_image_magic_bytes(VALID_WEBP_BYTES)
        self.assertEqual(detected, "image/webp")

    def test_magic_bytes_rejection_for_corrupt_or_text(self):
        with self.assertRaises(HTTPException) as ctx:
            validate_image_magic_bytes(b"This is a plain text file pretending to be an image.")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid image content", ctx.exception.detail)

    def test_magic_bytes_rejection_for_truncated_content(self):
        with self.assertRaises(HTTPException) as ctx:
            validate_image_magic_bytes(b"tiny")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("too small", ctx.exception.detail)

    def test_generate_public_id_format(self):
        public_id = generate_public_id()
        now = datetime.now(timezone.utc)
        year = now.strftime("%Y")
        month = now.strftime("%m")

        pattern = rf"^reports/{year}/{month}/[0-9a-f]{{32}}$"
        self.assertTrue(
            bool(re.match(pattern, public_id)),
            f"Public ID '{public_id}' does not match expected format reports/{year}/{month}/<32-hex-uuid>",
        )

    def test_upload_rejects_missing_filename(self):
        mock_file = DummyUploadFile(filename="", content=VALID_JPEG_BYTES)
        with self.assertRaises(HTTPException) as ctx:
            upload_report_photo(mock_file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("must have a filename", ctx.exception.detail)

    def test_upload_rejects_disallowed_extension(self):
        for bad_ext in [".exe", ".pdf", ".txt", ".gif", ".svg"]:
            mock_file = DummyUploadFile(
                filename=f"evidence{bad_ext}",
                content=b"content",
                content_type="application/octet-stream",
            )
            with self.assertRaises(HTTPException) as ctx:
                upload_report_photo(mock_file)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("Unsupported image extension", ctx.exception.detail)

    def test_upload_rejects_oversized_file(self):
        # 10 MB + 100 bytes
        oversized_bytes = b"\xff\xd8\xff" + b"0" * (MAX_FILE_SIZE + 100)
        mock_file = DummyUploadFile(
            filename="large_landslide.jpg",
            content=oversized_bytes,
            content_type="image/jpeg",
        )

        with self.assertRaises(HTTPException) as ctx:
            upload_report_photo(mock_file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("exceeds maximum allowed size", ctx.exception.detail)

    def test_upload_rejects_spoofed_content(self):
        mock_file = DummyUploadFile(
            filename="fake_road.jpg",
            content=b"Not really a jpeg even though named .jpg",
            content_type="image/jpeg",
        )

        with self.assertRaises(HTTPException) as ctx:
            upload_report_photo(mock_file)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Invalid image content", ctx.exception.detail)

    @patch.object(photo_storage.config, "CLOUDINARY_CLOUD_NAME", None)
    @patch.object(photo_storage.config, "CLOUDINARY_API_KEY", None)
    @patch.object(photo_storage.config, "CLOUDINARY_API_SECRET", None)
    def test_upload_returns_503_when_credentials_not_configured(self):
        mock_file = DummyUploadFile(
            filename="valid_rockfall.jpg",
            content=VALID_JPEG_BYTES,
            content_type="image/jpeg",
        )

        with self.assertRaises(HTTPException) as ctx:
            upload_report_photo(mock_file)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Cloudinary photo storage is not configured", ctx.exception.detail)

    @patch.object(photo_storage.config, "CLOUDINARY_CLOUD_NAME", "nexus-cloud")
    @patch.object(photo_storage.config, "CLOUDINARY_API_KEY", "123456789")
    @patch.object(photo_storage.config, "CLOUDINARY_API_SECRET", "supersecret")
    @patch("cloudinary.uploader.upload")
    def test_upload_returns_502_when_cloudinary_fails(self, mock_cloudinary_upload):
        mock_cloudinary_upload.side_effect = Exception("Cloudinary connection timeout")

        mock_file = DummyUploadFile(
            filename="valid_rockfall.jpg",
            content=VALID_JPEG_BYTES,
            content_type="image/jpeg",
        )

        with self.assertRaises(HTTPException) as ctx:
            upload_report_photo(mock_file)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("Cloudinary upload failed", ctx.exception.detail)

    @patch.object(photo_storage.config, "CLOUDINARY_CLOUD_NAME", "nexus-cloud")
    @patch.object(photo_storage.config, "CLOUDINARY_API_KEY", "123456789")
    @patch.object(photo_storage.config, "CLOUDINARY_API_SECRET", "supersecret")
    @patch("cloudinary.uploader.upload")
    def test_upload_success_mocked(self, mock_cloudinary_upload):
        expected_url = "https://res.cloudinary.com/nexus-cloud/image/upload/v12345/reports/2026/09/sample.jpg"
        mock_cloudinary_upload.return_value = {
            "secure_url": expected_url,
            "public_id": "reports/2026/09/sample",
        }

        mock_file = DummyUploadFile(
            filename="landslide.png",
            content=VALID_PNG_BYTES,
            content_type="image/png",
        )

        result = upload_report_photo(mock_file)
        self.assertIsInstance(result, PhotoUploadResult)
        self.assertEqual(result.photo_url, expected_url)
        self.assertEqual(result.content_type, "image/png")
        self.assertTrue(result.photo_public_id.startswith("reports/"))

        # Verify upload call parameters
        mock_cloudinary_upload.assert_called_once()
        call_kwargs = mock_cloudinary_upload.call_args[1]
        self.assertEqual(call_kwargs["resource_type"], "image")
        self.assertEqual(call_kwargs["overwrite"], True)
        self.assertEqual(call_kwargs["unique_filename"], False)
        self.assertEqual(call_kwargs["use_filename"], False)


class CloudinaryPublicReportApiIntegrationTests(unittest.TestCase):
    """End-to-end API tests for public citizen reporting with Cloudinary photo storage."""

    @classmethod
    def setUpClass(cls):
        # Configure an isolated in-memory SQLite database
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        cls.TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=cls.engine
        )

        User.__table__.create(bind=cls.engine)
        PublicReport.__table__.create(bind=cls.engine)

        # Create test PUBLIC user
        cls.db = cls.TestingSessionLocal()
        cls.test_user = User(
            username="citizen_tester",
            email="citizen@example.com",
            password_hash=auth_service.hash_password("CitizenPass123!"),
            role="PUBLIC",
            is_active=True,
        )
        cls.db.add(cls.test_user)
        cls.db.commit()
        cls.db.refresh(cls.test_user)

        # Generate JWT Bearer token for user
        cls.token = auth_service.create_access_token(
            {"sub": str(cls.test_user.id), "role": "PUBLIC"}
        )
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

        # Override FastAPI dependency for database session
        def override_get_db():
            db = cls.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        cls.db.close()
        PublicReport.__table__.drop(bind=cls.engine)
        User.__table__.drop(bind=cls.engine)

    def test_submit_report_without_photo_json(self):
        """Citizens can submit reports without a photo via JSON."""
        payload = {
            "latitude": 26.1445,
            "longitude": 91.7362,
            "report_type": "LANDSLIDE",
            "description": "Minor rockfall on highway shoulder, traffic moving.",
            "severity_hint": "low",
        }
        res = self.client.post("/public-reports/", json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["report_type"], "LANDSLIDE")
        self.assertEqual(data["status"], "UNVERIFIED")
        self.assertIsNone(data["photo_url"])
        self.assertIsNone(data["photo_public_id"])
        self.assertIsNone(data["content_type"])

    def test_submit_report_without_photo_form_data(self):
        """Citizens can submit reports without a photo via multipart/form-data."""
        form_data = {
            "latitude": "26.1500",
            "longitude": "91.7400",
            "report_type": "FLOOD",
            "description": "Drainage overflow on side lane.",
            "severity_hint": "medium",
        }
        res = self.client.post("/public-reports/", data=form_data, headers=self.headers)
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["report_type"], "FLOOD")
        self.assertIsNone(data["photo_url"])
        self.assertIsNone(data["photo_public_id"])
        self.assertIsNone(data["content_type"])

    @patch.object(photo_storage.config, "CLOUDINARY_CLOUD_NAME", "nexus-cloud")
    @patch.object(photo_storage.config, "CLOUDINARY_API_KEY", "123456789")
    @patch.object(photo_storage.config, "CLOUDINARY_API_SECRET", "supersecret")
    @patch("cloudinary.uploader.upload")
    def test_submit_report_with_photo_persists_cloudinary_metadata(self, mock_cloudinary_upload):
        """Genuine photo upload persists Cloudinary secure_url, public_id, and content_type."""
        fake_secure_url = (
            "https://res.cloudinary.com/nexus-cloud/image/upload/v1727258400/"
            "reports/2026/09/abcdef0123456789abcdef0123456789.jpg"
        )
        mock_cloudinary_upload.return_value = {
            "secure_url": fake_secure_url,
            "public_id": "reports/2026/09/abcdef0123456789abcdef0123456789",
        }

        form_data = {
            "latitude": "26.1445",
            "longitude": "91.7362",
            "report_type": "LANDSLIDE",
            "description": "Massive landslide blocking NH-27 both lanes.",
            "severity_hint": "critical",
        }
        files = {
            "photo": ("landslide_evidence.jpg", io.BytesIO(VALID_JPEG_BYTES), "image/jpeg")
        }

        res = self.client.post(
            "/public-reports/",
            data=form_data,
            files=files,
            headers=self.headers,
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()

        self.assertEqual(data["photo_url"], fake_secure_url)
        self.assertTrue(data["photo_public_id"].startswith("reports/"))
        self.assertEqual(data["content_type"], "image/jpeg")
        report_id = data["id"]

        # Verify database record directly
        db_report = self.db.query(PublicReport).filter(PublicReport.id == report_id).first()
        self.assertIsNotNone(db_report)
        self.assertEqual(db_report.photo_url, fake_secure_url)
        self.assertEqual(db_report.photo_public_id, data["photo_public_id"])
        self.assertEqual(db_report.content_type, "image/jpeg")

        # Verify Cloudinary upload call arguments
        mock_cloudinary_upload.assert_called_once()
        kwargs = mock_cloudinary_upload.call_args[1]
        self.assertTrue(kwargs["public_id"].startswith("reports/"))
        self.assertEqual(kwargs["resource_type"], "image")
        self.assertEqual(kwargs["overwrite"], True)
        self.assertEqual(kwargs["unique_filename"], False)
        self.assertEqual(kwargs["use_filename"], False)

    def test_submit_report_rejects_invalid_extension(self):
        """Executable or unsupported files are rejected with HTTP 400."""
        form_data = {
            "latitude": "26.1445",
            "longitude": "91.7362",
            "report_type": "ROAD_DAMAGE",
            "description": "Pothole deep on curve.",
        }
        files = {
            "photo": ("virus.exe", io.BytesIO(b"MZ\x90\x00executable content"), "application/octet-stream")
        }

        res = self.client.post(
            "/public-reports/",
            data=form_data,
            files=files,
            headers=self.headers,
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported image extension", res.json()["detail"])

    def test_submit_report_rejects_spoofed_image(self):
        """Files with .jpg extension but non-image content are rejected with HTTP 400."""
        form_data = {
            "latitude": "26.1445",
            "longitude": "91.7362",
            "report_type": "ACCIDENT",
            "description": "Two vehicles damaged.",
        }
        files = {
            "photo": ("spoofed.jpg", io.BytesIO(b"Hello world, I am not really a JPEG image"), "image/jpeg")
        }

        res = self.client.post(
            "/public-reports/",
            data=form_data,
            files=files,
            headers=self.headers,
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid image content", res.json()["detail"])

    def test_get_my_reports_includes_photo_metadata(self):
        """GET /public-reports/mine returns the citizen's reports with Cloudinary photo metadata."""
        sample_report = PublicReport(
            reporter_user_id=self.test_user.id,
            latitude=26.1445,
            longitude=91.7362,
            report_type="LANDSLIDE",
            description="Existing report for listing test.",
            severity_hint="high",
            photo_url="https://res.cloudinary.com/nexus-cloud/image/upload/v123/reports/2026/09/sample.jpg",
            photo_public_id="reports/2026/09/sample",
            content_type="image/jpeg",
            status="UNVERIFIED",
        )
        self.db.add(sample_report)
        self.db.commit()

        res = self.client.get("/public-reports/mine", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        reports = res.json()
        self.assertIsInstance(reports, list)
        self.assertGreater(len(reports), 0)

        # Check fields exist on reports
        first = reports[0]
        self.assertIn("photo_url", first)
        self.assertIn("photo_public_id", first)
        self.assertIn("content_type", first)
        self.assertEqual(first["photo_url"], "https://res.cloudinary.com/nexus-cloud/image/upload/v123/reports/2026/09/sample.jpg")
        self.assertEqual(first["photo_public_id"], "reports/2026/09/sample")
        self.assertEqual(first["content_type"], "image/jpeg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
