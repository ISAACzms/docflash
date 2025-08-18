"""
Adaptive Metadata Enhancement that infers domain and semantics from content and schema
No fixed document types - everything learned from actual data
Supports configurable richness levels: NONE, LOW, MEDIUM, HIGH
"""

import dspy
from typing import Dict, List, Any, Optional
import json
import asyncio
from collections import defaultdict, Counter
from enum import Enum


class MetadataRichness(Enum):
    """Metadata enhancement richness levels"""
    NONE = "none"      # No enhancement - keep existing attributes only
    LOW = "low"        # Basic entity types only
    MEDIUM = "medium"  # + Domain classification and functional roles  
    HIGH = "high"      # + Full semantic analysis and relationships


# DSPy Signatures for different richness levels

class BasicEntityClassification(dspy.Signature):
    """LOW richness: Basic entity type classification only"""
    
    entity_class = dspy.InputField(desc="The extraction class name")
    entity_text = dspy.InputField(desc="The extracted text value")
    context_hints = dspy.InputField(desc="Nearby entities for context")
    
    entity_type = dspy.OutputField(desc="Basic type: person, organization, amount, date, location, product, concept, etc.")


class EntityWithRelationships(dspy.Signature):
    """MEDIUM richness: Entity types + related entities"""
    
    entity_class = dspy.InputField(desc="The extraction class name")
    entity_text = dspy.InputField(desc="The extracted text value")
    peer_entities = dspy.InputField(desc="Other entities in same document")
    document_context = dspy.InputField(desc="Document type hints from schema")
    
    entity_type = dspy.OutputField(desc="Basic entity type (person, organization, amount, date, location, etc.)")
    related_entities = dspy.OutputField(desc="List of related entity classes that appear nearby or connect to this entity")


class ContentBasedDomainInference(dspy.Signature):
    """HIGH richness: Infer document domain and semantic context from content and extraction schema"""
    
    # Input - what we observe
    extraction_classes = dspy.InputField(desc="List of entity classes being extracted (e.g., ['client_name', 'medication', 'invoice_amount'])")
    sample_extracted_texts = dspy.InputField(desc="Sample of actual extracted text values")
    document_sections = dspy.InputField(desc="Document sections where entities were found")
    schema_description = dspy.InputField(desc="Schema description or name if available")
    
    # Output - what we infer
    domain_category = dspy.OutputField(desc="Inferred domain (e.g., 'healthcare', 'legal_services', 'financial_transactions')")
    semantic_context = dspy.OutputField(desc="Specific context within domain (e.g., 'patient_records', 'service_contracts', 'invoicing')")
    entity_nature = dspy.OutputField(desc="Nature of entities in this domain (e.g., 'people_organizations_money', 'patients_treatments', 'companies_amounts_dates')")


class SemanticAnalysisGenerator(dspy.Signature):
    """HIGH richness: Full semantic analysis with relationships"""
    
    entity_class = dspy.InputField(desc="The extraction class")
    entity_text = dspy.InputField(desc="The extracted text")
    peer_entities = dspy.InputField(desc="Other entities in same document")
    document_context = dspy.InputField(desc="Document type and context")
    inferred_domain = dspy.InputField(desc="Domain inferred from content analysis")
    semantic_context = dspy.InputField(desc="Specific semantic context")
    
    entity_type = dspy.OutputField(desc="Basic entity type")
    related_entities = dspy.OutputField(desc="List of related entity classes")
    relationship_types = dspy.OutputField(desc="Types of relationships with other entities (e.g., 'receives_treatment', 'owns_property')")
    semantic_context_output = dspy.OutputField(desc="Specific context this entity appears in (e.g., 'patient_care', 'financial_transactions')")


class AdaptiveRelationshipInference(dspy.Signature):
    """Infer relationships based on learned domain patterns"""
    
    entity1_class = dspy.InputField(desc="First entity class")
    entity1_text = dspy.InputField(desc="First entity text") 
    entity2_class = dspy.InputField(desc="Second entity class")
    entity2_text = dspy.InputField(desc="Second entity text")
    inferred_domain = dspy.InputField(desc="Domain context")
    semantic_context = dspy.InputField(desc="Specific semantic context")
    proximity_info = dspy.InputField(desc="How close the entities are")
    
    relationship_exists = dspy.OutputField(desc="Boolean: does a meaningful relationship exist")
    relationship_type = dspy.OutputField(desc="Type of relationship appropriate for this domain")
    relationship_strength = dspy.OutputField(desc="How strong/important this relationship is")


class AdaptiveMetadataEnhancer:
    """Fully adaptive metadata enhancer with configurable richness levels"""
    
    def __init__(self, richness_level: MetadataRichness = MetadataRichness.MEDIUM):
        self.richness_level = richness_level
        
        # Check DSPy availability
        try:
            if dspy.settings.lm is None:
                print(f"⚠️ DSPy LM not configured, richness level forced to NONE")
                self.use_dspy = False
                self.richness_level = MetadataRichness.NONE
            else:
                self.use_dspy = True
        except:
            self.use_dspy = False
            self.richness_level = MetadataRichness.NONE
        
        # Initialize DSPy predictors based on richness level
        if self.use_dspy:
            if self.richness_level in [MetadataRichness.LOW, MetadataRichness.MEDIUM, MetadataRichness.HIGH]:
                self.basic_classifier = dspy.Predict(BasicEntityClassification)
            
            if self.richness_level in [MetadataRichness.MEDIUM, MetadataRichness.HIGH]:
                self.relationship_classifier = dspy.Predict(EntityWithRelationships)
            
            if self.richness_level == MetadataRichness.HIGH:
                self.domain_inferrer = dspy.Predict(ContentBasedDomainInference)
                self.semantic_generator = dspy.Predict(SemanticAnalysisGenerator)
                self.relationship_inferrer = dspy.Predict(AdaptiveRelationshipInference)
            else:
                self.domain_inferrer = None
                self.semantic_generator = None 
                self.relationship_inferrer = None
        else:
            # No DSPy available
            self.basic_classifier = None
            self.relationship_classifier = None
            self.domain_inferrer = None
            self.semantic_generator = None 
            self.relationship_inferrer = None
    
    async def enhance_extractions(self, extractions: List[Dict], document_schema: Dict = None) -> List[Dict]:
        """Enhance extractions with adaptive semantic metadata at specified richness level"""
        
        if self.richness_level == MetadataRichness.NONE:
            # No enhancement - return as-is
            return extractions
        
        if self.richness_level == MetadataRichness.LOW:
            return await self._enhance_low_richness(extractions)
        
        elif self.richness_level == MetadataRichness.MEDIUM:
            return await self._enhance_medium_richness(extractions, document_schema)
        
        elif self.richness_level == MetadataRichness.HIGH:
            return await self._enhance_high_richness(extractions, document_schema)
        
        return extractions
    
    async def _enhance_low_richness(self, extractions: List[Dict]) -> List[Dict]:
        """LOW richness: Basic entity type classification only"""
        
        if not self.basic_classifier:
            return extractions
        
        enhanced = []
        for extraction in extractions:
            enhanced_extraction = extraction.copy()
            current_attrs = extraction.get('attributes', {})
            
            # Keep existing attributes
            enhanced_attrs = current_attrs.copy()
            
            try:
                # Get context from peer entities
                peer_entities = [ext.get('extraction_class', '') for ext in extractions if ext != extraction]
                context_hints = ', '.join(peer_entities[:3])  # First 3 for context
                
                # Basic entity classification
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.basic_classifier(
                        entity_class=extraction.get('extraction_class', ''),
                        entity_text=extraction.get('extraction_text', ''),
                        context_hints=context_hints
                    )
                )
                
                enhanced_attrs['entity_type'] = result.entity_type
                
            except Exception as e:
                print(f"⚠️ Low richness enhancement failed for {extraction.get('extraction_class', '')}: {e}")
                enhanced_attrs['entity_type'] = 'unknown'
            
            enhanced_extraction['attributes'] = enhanced_attrs
            enhanced.append(enhanced_extraction)
        
        return enhanced
    
    async def _enhance_medium_richness(self, extractions: List[Dict], document_schema: Dict = None) -> List[Dict]:
        """MEDIUM richness: Entity types + related entities"""
        
        if not self.relationship_classifier:
            return await self._enhance_low_richness(extractions)
        
        enhanced = []
        for extraction in extractions:
            enhanced_extraction = extraction.copy()
            current_attrs = extraction.get('attributes', {})
            
            # Keep existing attributes
            enhanced_attrs = current_attrs.copy()
            
            try:
                # Get context from peer entities and schema
                peer_entities = [ext.get('extraction_class', '') for ext in extractions if ext != extraction]
                peer_context = ', '.join(peer_entities)
                
                document_context = document_schema.get('description', '') if document_schema else ''
                if not document_context:
                    document_context = document_schema.get('name', '') if document_schema else ''
                
                # Entity with relationships classification
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.relationship_classifier(
                        entity_class=extraction.get('extraction_class', ''),
                        entity_text=extraction.get('extraction_text', ''),
                        peer_entities=peer_context,
                        document_context=document_context
                    )
                )
                
                enhanced_attrs.update({
                    'entity_type': result.entity_type,
                    'related_entities': result.related_entities
                })
                
            except Exception as e:
                print(f"⚠️ Medium richness enhancement failed for {extraction.get('extraction_class', '')}: {e}")
                # Fallback to low richness
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.basic_classifier(
                            entity_class=extraction.get('extraction_class', ''),
                            entity_text=extraction.get('extraction_text', ''),
                            context_hints=', '.join(peer_entities[:3])
                        )
                    )
                    enhanced_attrs['entity_type'] = result.entity_type
                    enhanced_attrs['related_entities'] = []  # Empty list as fallback
                except:
                    enhanced_attrs['entity_type'] = 'unknown'
                    enhanced_attrs['related_entities'] = []
            
            enhanced_extraction['attributes'] = enhanced_attrs
            enhanced.append(enhanced_extraction)
        
        return enhanced
    
    async def _enhance_high_richness(self, extractions: List[Dict], document_schema: Dict = None) -> List[Dict]:
        """HIGH richness: Full semantic analysis (original behavior)"""
        
        # Step 1: Infer domain context from actual content
        domain_context = await self._infer_domain_context(extractions, document_schema)
        
        # Step 2: Generate semantic attributes using inferred context (parallel processing)
        enhancement_tasks = [
            self._enhance_single_extraction_high(extraction, extractions, domain_context)
            for extraction in extractions
        ]
        enhanced_attributes = await asyncio.gather(*enhancement_tasks)
        
        # Combine enhanced attributes with original extractions
        enhanced = []
        for extraction, attrs in zip(extractions, enhanced_attributes):
            enhanced_extraction = extraction.copy()
            enhanced_extraction['attributes'] = attrs
            enhanced.append(enhanced_extraction)
        
        # Step 3: Infer relationships using learned domain patterns
        enhanced = await self._add_adaptive_relationships(enhanced, domain_context)
        
        return enhanced
    
    async def _infer_domain_context(self, extractions: List[Dict], document_schema: Dict = None) -> Dict:
        """Infer domain and semantic context from actual content"""
        
        # Gather content indicators
        extraction_classes = [ext.get('extraction_class', '') for ext in extractions]
        sample_texts = [ext.get('extraction_text', '') for ext in extractions if ext.get('extraction_text')]
        sections = [ext.get('attributes', {}).get('section', '') for ext in extractions]
        schema_desc = document_schema.get('description', '') if document_schema else ''
        
        if self.use_dspy and self.domain_inferrer:
            try:
                # Use DSPy to infer domain from content (run in executor for async)
                domain_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.domain_inferrer(
                        extraction_classes=str(extraction_classes),
                        sample_extracted_texts=str(sample_texts[:5]),  # First 5 samples
                        document_sections=str(list(set(sections))),
                        schema_description=schema_desc
                    )
                )
                
                return {
                    'domain_category': domain_result.domain_category,
                    'semantic_context': domain_result.semantic_context,
                    'entity_nature': domain_result.entity_nature,
                    'inference_method': 'dspy_content_analysis'
                }
                
            except Exception as e:
                print(f"⚠️ Domain inference failed: {e}")
                return self._fallback_domain_inference(extraction_classes, sample_texts, schema_desc)
        else:
            return self._smart_content_analysis(extraction_classes, sample_texts, sections, schema_desc)
    
    def _smart_content_analysis(self, classes: List[str], texts: List[str], sections: List[str], schema_desc: str) -> Dict:
        """Smart content analysis without LLM"""
        
        # Analyze extraction classes for patterns
        class_indicators = {
            'healthcare': ['patient', 'medication', 'diagnosis', 'dosage', 'doctor', 'treatment'],
            'legal': ['client', 'contract', 'agreement', 'party', 'terms', 'liability'],
            'financial': ['amount', 'invoice', 'payment', 'account', 'balance', 'transaction'],
            'academic': ['student', 'course', 'grade', 'professor', 'university', 'degree'],
            'real_estate': ['property', 'address', 'buyer', 'seller', 'price', 'listing'],
            'employment': ['employee', 'salary', 'position', 'department', 'manager', 'benefits']
        }
        
        # Analyze text content patterns
        text_patterns = {
            'financial': ['$', '€', '£', 'USD', 'payment', 'invoice', 'amount'],
            'healthcare': ['mg', 'ml', 'daily', 'twice', 'prescription', 'diagnosis'],
            'legal': ['agreement', 'contract', 'party', 'hereby', 'whereas', 'pursuant'],
            'temporal': ['date', 'time', 'month', 'year', 'deadline', 'schedule'],
            'personal': ['name', 'address', 'phone', 'email', 'contact']
        }
        
        # Score domains based on class names
        domain_scores = defaultdict(int)
        for class_name in classes:
            class_lower = class_name.lower()
            for domain, indicators in class_indicators.items():
                for indicator in indicators:
                    if indicator in class_lower:
                        domain_scores[domain] += 2
        
        # Score based on text content
        all_text = ' '.join(texts).lower()
        for domain, patterns in text_patterns.items():
            for pattern in patterns:
                if pattern in all_text:
                    domain_scores[domain] += 1
        
        # Score based on schema description
        if schema_desc:
            schema_lower = schema_desc.lower()
            for domain, indicators in class_indicators.items():
                for indicator in indicators:
                    if indicator in schema_lower:
                        domain_scores[domain] += 3
        
        # Determine primary domain
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            primary_domain = 'business_document'
        
        # Infer semantic context
        if 'contract' in all_text or 'agreement' in all_text:
            context = 'contractual_relationships'
        elif 'patient' in all_text or 'medication' in all_text:
            context = 'patient_care'
        elif 'invoice' in all_text or 'payment' in all_text:
            context = 'financial_transactions'
        else:
            context = 'information_extraction'
        
        # Analyze entity nature
        entity_types = []
        if any('name' in c for c in classes):
            entity_types.append('people_organizations')
        if any(c in ['amount', 'value', 'price', 'cost'] for c in classes):
            entity_types.append('monetary_values')
        if any('date' in c or 'time' in c for c in classes):
            entity_types.append('temporal_references')
        
        entity_nature = '_'.join(entity_types) if entity_types else 'general_entities'
        
        return {
            'domain_category': primary_domain,
            'semantic_context': context,
            'entity_nature': entity_nature,
            'inference_method': 'smart_content_analysis',
            'confidence_score': max(domain_scores.values()) if domain_scores else 1
        }
    
    async def _enhance_single_extraction_high(self, extraction: Dict, all_extractions: List[Dict], domain_context: Dict) -> Dict:
        """Generate full semantic attributes for HIGH richness"""
        
        current_attrs = extraction.get('attributes', {})
        enhanced_attrs = current_attrs.copy()
        
        entity_class = extraction.get('extraction_class', '')
        entity_text = extraction.get('extraction_text', '')
        
        # Get peer entities for context
        peer_entities = [ext.get('extraction_class') for ext in all_extractions if ext != extraction]
        document_context = f"Domain: {domain_context['domain_category']}, Context: {domain_context['semantic_context']}"
        
        if self.use_dspy and self.semantic_generator:
            try:
                # Run DSPy semantic generation in executor for async
                semantic_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.semantic_generator(
                        entity_class=entity_class,
                        entity_text=entity_text,
                        peer_entities=str(peer_entities),
                        document_context=document_context,
                        inferred_domain=domain_context['domain_category'],
                        semantic_context=domain_context['semantic_context']
                    )
                )
                
                enhanced_attrs.update({
                    'entity_type': semantic_result.entity_type,
                    'related_entities': semantic_result.related_entities,
                    'relationship_types': semantic_result.relationship_types,
                    'semantic_context': semantic_result.semantic_context_output
                })
                
            except Exception as e:
                print(f"⚠️ HIGH richness enhancement failed for {entity_class}: {e}")
                enhanced_attrs.update(self._high_fallback_attributes(entity_class, entity_text, domain_context))
        else:
            enhanced_attrs.update(self._high_fallback_attributes(entity_class, entity_text, domain_context))
        
        # Remove old static attributes
        enhanced_attrs.pop('mode', None)
        enhanced_attrs.pop('dspy_generated', None)
        
        return enhanced_attrs
    
    def _high_fallback_attributes(self, entity_class: str, entity_text: str, domain_context: Dict) -> Dict:
        """Generate HIGH richness fallback attributes"""
        
        # Basic entity type classification
        if 'name' in entity_class.lower():
            entity_type = 'person'
        elif any(term in entity_class.lower() for term in ['company', 'organization', 'corp']):
            entity_type = 'organization'
        elif any(term in entity_class.lower() for term in ['amount', 'value', 'cost', 'price']):
            entity_type = 'amount'
        elif any(term in entity_class.lower() for term in ['date', 'time']):
            entity_type = 'date'
        elif any(term in entity_class.lower() for term in ['address', 'location']):
            entity_type = 'location'
        else:
            entity_type = 'concept'
        
        # Simple related entities (just neighboring entity classes)
        related_entities = []
        
        # Basic relationship types
        relationship_types = ['related_to']
        
        # Simple semantic context
        context = domain_context.get('semantic_context', 'information_extraction')
        
        return {
            'entity_type': entity_type,
            'related_entities': related_entities,
            'relationship_types': relationship_types,
            'semantic_context': context
        }
    
    async def _add_adaptive_relationships(self, extractions: List[Dict], domain_context: Dict) -> List[Dict]:
        """Add relationships using adaptive inference"""
        
        for i, extraction in enumerate(extractions):
            related_entities = []
            relationship_types = []
            
            for j, other_extraction in enumerate(extractions):
                if i == j:
                    continue
                
                proximity = self._assess_proximity(extraction, other_extraction)
                
                if self.use_dspy and self.relationship_inferrer:
                    try:
                        # Run DSPy relationship inference in executor for async
                        rel_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.relationship_inferrer(
                                entity1_class=extraction.get('extraction_class', ''),
                                entity1_text=extraction.get('extraction_text', ''),
                                entity2_class=other_extraction.get('extraction_class', ''),
                                entity2_text=other_extraction.get('extraction_text', ''),
                                inferred_domain=domain_context['domain_category'],
                                semantic_context=domain_context['semantic_context'],
                                proximity_info=proximity
                            )
                        )
                        
                        if rel_result.relationship_exists.lower() == 'true':
                            related_entities.append(other_extraction.get('extraction_class'))
                            relationship_types.append(rel_result.relationship_type)
                            
                    except Exception as e:
                        print(f"⚠️ Relationship inference failed: {e}")
                        # Fallback to proximity-based
                        if proximity in ['same_section', 'nearby_text']:
                            related_entities.append(other_extraction.get('extraction_class'))
                else:
                    # Adaptive rule-based relationships
                    if proximity in ['same_section', 'nearby_text']:
                        related_entities.append(other_extraction.get('extraction_class'))
                        rel_type = self._infer_adaptive_relationship(
                            extraction.get('extraction_class'),
                            other_extraction.get('extraction_class'),
                            domain_context
                        )
                        if rel_type:
                            relationship_types.append(rel_type)
            
            extraction['attributes']['related_entities'] = list(set(related_entities))
            extraction['attributes']['relationship_types'] = list(set(relationship_types))
        
        return extractions
    
    def _infer_adaptive_relationship(self, class1: str, class2: str, domain_context: Dict) -> Optional[str]:
        """Adaptively infer relationship based on learned domain"""
        
        domain = domain_context['domain_category']
        
        # Generate relationship based on semantic understanding
        if 'name' in class1 and 'name' in class2:
            return 'associated_with'
        elif 'name' in class1 and 'amount' in class2:
            if domain == 'financial':
                return 'owes_amount' if 'client' in class1 else 'receives_amount'
            else:
                return 'has_value'
        elif 'medication' in class1 and 'dosage' in class2:
            return 'has_dosage'
        elif 'medication' in class1 and 'name' in class2:
            return 'prescribed_to'
        elif any(term in class1 for term in ['contract', 'agreement']) and 'name' in class2:
            return 'party_to'
        else:
            return 'related_to'
    
    def _assess_proximity(self, extraction1: Dict, extraction2: Dict) -> str:
        """Assess proximity between extractions"""
        section1 = extraction1.get('attributes', {}).get('section', '')
        section2 = extraction2.get('attributes', {}).get('section', '')
        
        if section1 == section2 and section1:
            return 'same_section'
        
        interval1 = extraction1.get('char_interval')
        interval2 = extraction2.get('char_interval')
        
        if interval1 and interval2:
            pos1 = getattr(interval1, 'start_pos', None)
            pos2 = getattr(interval2, 'start_pos', None)
            if pos1 and pos2 and abs(pos1 - pos2) < 200:
                return 'nearby_text'
        
        return 'distant'
    
    def _fallback_domain_inference(self, classes: List[str], texts: List[str], schema_desc: str) -> Dict:
        """Basic fallback domain inference"""
        return {
            'domain_category': 'general_document',
            'semantic_context': 'information_extraction', 
            'entity_nature': 'mixed_entities',
            'inference_method': 'basic_fallback',
            'confidence_score': 0.5
        }


async def enhance_extraction_metadata_adaptive(
    extractions: List[Dict], 
    document_schema: Dict = None, 
    richness_level: MetadataRichness = MetadataRichness.MEDIUM
) -> List[Dict]:
    """Main function for adaptive metadata enhancement with configurable richness (async)"""
    print(f"🧠 [ADAPTIVE] Starting {richness_level.value} richness enhancement for {len(extractions)} extractions")
    
    enhancer = AdaptiveMetadataEnhancer(richness_level)
    enhanced = await enhancer.enhance_extractions(extractions, document_schema)
    
    print(f"🧠 [ADAPTIVE] Enhanced {len(enhanced)} extractions at {richness_level.value} level")
    if enhanced:
        sample_attrs = enhanced[0].get('attributes', {})
        print(f"🧠 [ADAPTIVE] Sample enhanced attributes: {list(sample_attrs.keys())}")
    return enhanced