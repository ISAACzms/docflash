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
    
    def _make_template_key(self, document_class: str, domain_name: str) -> str:
        """Create a composite key for template storage"""
        return f"{domain_name}:{document_class}"
    
    def _parse_template_key(self, key: str) -> tuple[str, str]:
        """Parse a composite key back into domain_name and document_class"""
        if ":" in key:
            domain_name, document_class = key.split(":", 1)
            return domain_name, document_class
        else:
            # Legacy key (just document_class)
            return None, key

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

        # Use composite key (domain:document_class) for unique identification
        template_key = self._make_template_key(document_class, domain_name)
        self.templates[template_key] = template
        self._save_db()

        print(f"✅ Registered template for '{document_class}' in domain '{domain_name}' with ID: {template_id}")
        return template_id

    def get_template(self, document_class: str, domain_name: Optional[str] = None) -> Optional[Dict]:
        """Get template by document class and domain (with backward compatibility)"""
        if domain_name is not None:
            # Use composite key for domain-specific lookup
            template_key = self._make_template_key(document_class, domain_name)
            return self.templates.get(template_key)
        else:
            # Backward compatibility: try to find template by document_class only
            # First try legacy key (just document_class)
            template = self.templates.get(document_class)
            if template:
                return template
            
            # If not found, search through all templates for matching document_class
            for key, template in self.templates.items():
                if ":" in key:
                    # Parse composite key
                    _, stored_doc_class = self._parse_template_key(key)
                    if stored_doc_class == document_class:
                        return template
            
            return None

    def list_templates(self) -> List[Dict]:
        """List all registered templates"""
        return list(self.templates.values())

    def update_template_usage(self, document_class: str, domain_name: Optional[str] = None):
        """Update usage statistics for a template"""
        if domain_name is not None:
            template_key = self._make_template_key(document_class, domain_name)
            if template_key in self.templates:
                self.templates[template_key]["last_used"] = datetime.now().isoformat()
                self.templates[template_key]["usage_count"] += 1
                self._save_db()
        else:
            # Backward compatibility: find the template by document_class
            template_key = None
            # First try legacy key
            if document_class in self.templates:
                template_key = document_class
            else:
                # Search through composite keys
                for key, template in self.templates.items():
                    if ":" in key:
                        _, stored_doc_class = self._parse_template_key(key)
                        if stored_doc_class == document_class:
                            template_key = key
                            break
            
            if template_key:
                self.templates[template_key]["last_used"] = datetime.now().isoformat()
                self.templates[template_key]["usage_count"] += 1
                self._save_db()

    def delete_template(self, document_class: str, domain_name: Optional[str] = None) -> bool:
        """Delete a template"""
        if domain_name is not None:
            template_key = self._make_template_key(document_class, domain_name)
            if template_key in self.templates:
                del self.templates[template_key]
                self._save_db()
                return True
        else:
            # Backward compatibility: find and delete by document_class
            template_key = None
            # First try legacy key
            if document_class in self.templates:
                template_key = document_class
            else:
                # Search through composite keys
                for key in self.templates.keys():
                    if ":" in key:
                        _, stored_doc_class = self._parse_template_key(key)
                        if stored_doc_class == document_class:
                            template_key = key
                            break
            
            if template_key:
                del self.templates[template_key]
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
        template_key = self._make_template_key(document_class, domain_name)
        if template_key not in self.templates:
            return False
            
        # Get existing template to preserve metadata
        existing_template = self.templates[template_key]
        
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
        
        self.templates[template_key] = updated_template
        self._save_db()
        
        print(f"✅ Updated template for '{document_class}' in domain '{domain_name}'")
        return True

    def template_exists(self, document_class: str, domain_name: Optional[str] = None) -> bool:
        """Check if template exists"""
        if domain_name is not None:
            template_key = self._make_template_key(document_class, domain_name)
            return template_key in self.templates
        else:
            # Backward compatibility: check if any template with this document_class exists
            # First try legacy key
            if document_class in self.templates:
                return True
            
            # Search through composite keys
            for key in self.templates.keys():
                if ":" in key:
                    _, stored_doc_class = self._parse_template_key(key)
                    if stored_doc_class == document_class:
                        return True
            
            return False

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