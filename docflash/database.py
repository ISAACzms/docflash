#!/usr/bin/env python3
"""
Simple JSON-based database for storing document templates
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class DocumentTemplateDB:
    def __init__(self, db_file="document_templates.json"):
        self.db_file = db_file
        self.templates = self._load_db()

    def _load_db(self) -> Dict:
        """Load templates from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_db(self):
        """Save templates to JSON file"""
        try:
            with open(self.db_file, "w") as f:
                json.dump(self.templates, f, indent=2, default=str)
        except IOError as e:
            print(f"Error saving database: {e}")

    def register_template(
        self,
        document_class: str,
        extraction_schema: List[Dict],
        prompt_description: str,
        examples: List[Dict],
        output_schema: Dict,
    ) -> str:
        """Register a new document template"""

        template_id = str(uuid.uuid4())[:8]

        template = {
            "template_id": template_id,
            "document_class": document_class,
            "extraction_schema": extraction_schema,
            "prompt_description": prompt_description,
            "examples": examples,
            "output_schema": output_schema,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
        }

        # Use document_class as key, but keep template_id for reference
        self.templates[document_class] = template
        self._save_db()

        print(f"✅ Registered template for '{document_class}' with ID: {template_id}")
        return template_id

    def get_template(self, document_class: str) -> Optional[Dict]:
        """Get template by document class"""
        return self.templates.get(document_class)

    def list_templates(self) -> List[Dict]:
        """List all registered templates"""
        return list(self.templates.values())

    def update_template_usage(self, document_class: str):
        """Update usage statistics for a template"""
        if document_class in self.templates:
            self.templates[document_class]["last_used"] = datetime.now().isoformat()
            self.templates[document_class]["usage_count"] += 1
            self._save_db()

    def delete_template(self, document_class: str) -> bool:
        """Delete a template"""
        if document_class in self.templates:
            del self.templates[document_class]
            self._save_db()
            return True
        return False

    def template_exists(self, document_class: str) -> bool:
        """Check if template exists"""
        return document_class in self.templates


# Global database instance
db = DocumentTemplateDB()