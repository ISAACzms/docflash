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
    def __init__(self, db_file="document_templates.json", domains_file="document_domains.json"):
        self.db_file = db_file
        self.domains_file = domains_file
        self.templates = self._load_db()
        self.domains = self._load_domains()

    def _load_db(self) -> Dict:
        """Load templates from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _load_domains(self) -> Dict:
        """Load domains from JSON file"""
        if os.path.exists(self.domains_file):
            try:
                with open(self.domains_file, "r") as f:
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

    def _save_domains(self):
        """Save domains to JSON file"""
        try:
            with open(self.domains_file, "w") as f:
                json.dump(self.domains, f, indent=2, default=str)
        except IOError as e:
            print(f"Error saving domains: {e}")

    def register_template(
        self,
        document_class: str,
        extraction_schema: List[Dict],
        prompt_description: str,
        examples: List[Dict],
        output_schema: Dict,
        domain_name: str,
        additional_instructions: str = "",
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
            "domain_name": domain_name,
            "additional_instructions": additional_instructions,
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

    def update_template(
        self,
        document_class: str,
        extraction_schema: List[Dict],
        prompt_description: str,
        examples: List[Dict],
        output_schema: Dict,
        domain_name: str,
        additional_instructions: str = "",
    ) -> bool:
        """Update an existing document template (preserves metadata)"""
        if document_class not in self.templates:
            return False
            
        # Get existing template to preserve metadata
        existing_template = self.templates[document_class]
        
        # Update template data while preserving metadata
        updated_template = {
            "template_id": existing_template.get("template_id"),
            "document_class": document_class,
            "extraction_schema": extraction_schema,
            "prompt_description": prompt_description,
            "examples": examples,
            "output_schema": output_schema,
            "domain_name": domain_name,
            "additional_instructions": additional_instructions,
            "created_at": existing_template.get("created_at"),
            "last_used": existing_template.get("last_used"),
            "usage_count": existing_template.get("usage_count", 0),
            "updated_at": datetime.now().isoformat(),
        }
        
        self.templates[document_class] = updated_template
        self._save_db()
        
        print(f"✅ Updated template for '{document_class}'")
        return True

    def template_exists(self, document_class: str) -> bool:
        """Check if template exists"""
        return document_class in self.templates

    # Domain Management Methods
    def register_domain(self, domain_name: str, description: str = "") -> str:
        """Register a new domain"""
        domain_id = str(uuid.uuid4())[:8]
        
        domain = {
            "domain_id": domain_id,
            "domain_name": domain_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "document_classes": [],
            "total_templates": 0
        }
        
        self.domains[domain_name] = domain
        self._save_domains()
        
        print(f"✅ Registered domain '{domain_name}' with ID: {domain_id}")
        return domain_id

    def get_domain(self, domain_name: str) -> Optional[Dict]:
        """Get domain by name"""
        return self.domains.get(domain_name)

    def list_domains(self) -> List[Dict]:
        """List all registered domains"""
        return list(self.domains.values())

    def update_domain(self, domain_name: str, description: str) -> bool:
        """Update domain description"""
        if domain_name in self.domains:
            self.domains[domain_name]["description"] = description
            self._save_domains()
            return True
        return False

    def delete_domain(self, domain_name: str) -> bool:
        """Delete a domain (only if no templates are assigned to it)"""
        if domain_name in self.domains:
            # Check if any templates are assigned to this domain
            templates_in_domain = [
                t for t in self.templates.values() 
                if t.get("domain_name") == domain_name
            ]
            
            if templates_in_domain:
                print(f"❌ Cannot delete domain '{domain_name}' - {len(templates_in_domain)} templates are assigned to it")
                return False
            
            del self.domains[domain_name]
            self._save_domains()
            print(f"✅ Deleted domain '{domain_name}'")
            return True
        return False

    def domain_exists(self, domain_name: str) -> bool:
        """Check if domain exists"""
        return domain_name in self.domains

    def get_templates_by_domain(self, domain_name: str) -> List[Dict]:
        """Get all templates assigned to a specific domain"""
        return [
            template for template in self.templates.values()
            if template.get("domain_name") == domain_name
        ]

    def update_domain_statistics(self):
        """Update domain statistics (document class counts)"""
        # Reset all domain stats
        for domain in self.domains.values():
            domain["document_classes"] = []
            domain["total_templates"] = 0
        
        # Count templates per domain
        for template in self.templates.values():
            domain_name = template.get("domain_name")
            if domain_name and domain_name in self.domains:
                self.domains[domain_name]["document_classes"].append(template["document_class"])
                self.domains[domain_name]["total_templates"] += 1
        
        self._save_domains()


# Global database instance
db = DocumentTemplateDB()