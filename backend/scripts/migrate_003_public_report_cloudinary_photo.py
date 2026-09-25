#!/usr/bin/env python3
"""
NEXUS-NER Migration 003: Public Report Cloudinary Photo Metadata
===============================================================
Adds `photo_public_id` and `content_type` columns to `public_reports` table
and expands `photo_url` length to support Cloudinary durable storage URLs.

Usage:
  python backend/scripts/migrate_003_public_report_cloudinary_photo.py
  python backend/scripts/migrate_003_public_report_cloudinary_photo.py --downgrade
"""

import argparse
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import text
from app.database import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nexus_ner.migration_003")


def upgrade():
    logger.info("Executing Migration 003: UPGRADE (adding Cloudinary photo metadata)...")
    try:
        with engine.begin() as conn:
            # 1. Expand photo_url if needed
            conn.execute(
                text(
                    """
                    ALTER TABLE public_reports
                        ALTER COLUMN photo_url TYPE VARCHAR(512);
                    """
                )
            )

            # 2. Add photo_public_id column
            conn.execute(
                text(
                    """
                    ALTER TABLE public_reports
                        ADD COLUMN IF NOT EXISTS photo_public_id VARCHAR(255);
                    """
                )
            )

            # 3. Add content_type column
            conn.execute(
                text(
                    """
                    ALTER TABLE public_reports
                        ADD COLUMN IF NOT EXISTS content_type VARCHAR(50);
                    """
                )
            )

            # 4. Add index on photo_public_id
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_public_reports_photo_public_id
                        ON public_reports(photo_public_id);
                    """
                )
            )

        logger.info("Migration 003 UPGRADE completed successfully.")
    except Exception as err:
        logger.error("Migration 003 UPGRADE failed (is PostgreSQL running on port 5433?): %s", err)
        raise


def downgrade():
    logger.info("Executing Migration 003: DOWNGRADE (reverting Cloudinary photo metadata)...")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DROP INDEX IF EXISTS ix_public_reports_photo_public_id;
                ALTER TABLE public_reports DROP COLUMN IF EXISTS photo_public_id;
                ALTER TABLE public_reports DROP COLUMN IF EXISTS content_type;
                ALTER TABLE public_reports ALTER COLUMN photo_url TYPE VARCHAR(255);
                """
            )
        )
    logger.info("Migration 003 DOWNGRADE completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migration 003: Public Report Cloudinary Photo Metadata")
    parser.add_argument("--downgrade", action="store_true", help="Revert the migration")
    args = parser.parse_args()

    if args.downgrade:
        downgrade()
    else:
        upgrade()
