"""
Adaptive Metadata Enhancement that infers domain and semantics from content and schema
No fixed document types - everything learned from actual data
"""

import dspy
from typing import Dict, List, Any, Optional
import json
import asyncio
from collections import defaultdict, Counter


class ContentBasedDomainInference(dspy.Signature):
    """Infer document domain and semantic context from content and extraction schema"""
    
    # Input - what we observe
    extraction_classes = dspy.InputField(desc="List of entity classes being extracted (e.g., ['client_name', 'medication', 'invoice_amount'])")
    sample_extracted_texts = dspy.InputField(desc="Sample of actual extracted text values")
    document_sections = dspy.InputField(desc="Document sections where entities were found")
    schema_description = dspy.InputField(desc="Schema description or name if available")
    
    # Output - what we infer
    domain_category = dspy.OutputField(desc="Inferred domain (e.g., 'healthcare', 'legal_services', 'financial_transactions')")
    semantic_context = dspy.OutputField(desc="Specific context within domain (e.g., 'patient_records', 'service_contracts', 'invoicing')")
    entity_nature = dspy.OutputField(desc="Nature of entities in this domain (e.g., 'people_organizations_money', 'patients_treatments', 'companies_amounts_dates')")


class ContextAwareSemanticGenerator(dspy.Signature):
    """Generate semantic attributes based on inferred domain context"""
    
    # Input context (no fixed domains)
    entity_class = dspy.InputField(desc="The extraction class")
    entity_text = dspy.InputField(desc="The extracted text")
    inferred_domain = dspy.InputField(desc="Domain inferred from content analysis")
    semantic_context = dspy.InputField(desc="Specific semantic context")
    peer_entities = dspy.InputField(desc="Other entities in same schema/document")
    document_section = dspy.InputField(desc="Section where found")
    
    # Output - context-aware semantics
    functional_role = dspy.OutputField(desc="What role this entity plays in the domain context")
    semantic_category = dspy.OutputField(desc="Category that makes sense for this domain")
    contextual_significance = dspy.OutputField(desc="Why this entity matters in this specific context")
    relationship_hints = dspy.OutputField(desc="Likely relationships with peer entities")


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
    """Fully adaptive metadata enhancer - learns everything from content"""
    
    def __init__(self):
        # Check DSPy availability
        try:
            if dspy.settings.lm is None:
                print("⚠️ DSPy LM not configured, using adaptive fallback mode")
                self.use_dspy = False
            else:
                self.use_dspy = True
        except:
            self.use_dspy = False
        
        if self.use_dspy:
            self.domain_inferrer = dspy.Predict(ContentBasedDomainInference)
            self.semantic_generator = dspy.Predict(ContextAwareSemanticGenerator)
            self.relationship_inferrer = dspy.Predict(AdaptiveRelationshipInference)
        else:
            self.domain_inferrer = None
            self.semantic_generator = None 
            self.relationship_inferrer = None
    
    async def enhance_extractions(self, extractions: List[Dict], document_schema: Dict = None) -> List[Dict]:
        """Enhance extractions with adaptive semantic metadata asynchronously"""
        
        # Step 1: Infer domain context from actual content
        domain_context = await self._infer_domain_context(extractions, document_schema)
        
        # Step 2: Generate semantic attributes using inferred context (parallel processing)
        enhancement_tasks = [
            self._enhance_single_extraction(extraction, extractions, domain_context)
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
    
    async def _enhance_single_extraction(self, extraction: Dict, all_extractions: List[Dict], domain_context: Dict) -> Dict:
        """Generate semantic attributes using inferred domain context"""
        
        current_attrs = extraction.get('attributes', {})
        enhanced_attrs = current_attrs.copy()
        
        # Keep useful existing attributes
        if 'page_number' in current_attrs:
            enhanced_attrs['page_number'] = current_attrs['page_number']
        
        entity_class = extraction.get('extraction_class', '')
        entity_text = extraction.get('extraction_text', '')
        document_section = current_attrs.get('section', '')
        
        # Get peer entities for context
        peer_entities = [ext.get('extraction_class') for ext in all_extractions if ext != extraction]
        
        if self.use_dspy and self.semantic_generator:
            try:
                # Run DSPy semantic generation in executor for async
                semantic_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.semantic_generator(
                        entity_class=entity_class,
                        entity_text=entity_text,
                        inferred_domain=domain_context['domain_category'],
                        semantic_context=domain_context['semantic_context'],
                        peer_entities=str(peer_entities),
                        document_section=document_section
                    )
                )
                
                enhanced_attrs.update({
                    'functional_role': semantic_result.functional_role,
                    'semantic_category': semantic_result.semantic_category,
                    'contextual_significance': semantic_result.contextual_significance,
                    'relationship_hints': semantic_result.relationship_hints,
                    'inferred_domain': domain_context['domain_category'],
                    'semantic_context': domain_context['semantic_context']
                })
                
            except Exception as e:
                print(f"⚠️ Semantic generation failed for {entity_class}: {e}")
                enhanced_attrs.update(self._adaptive_fallback_attributes(entity_class, entity_text, domain_context))
        else:
            enhanced_attrs.update(self._adaptive_fallback_attributes(entity_class, entity_text, domain_context))
        
        # Remove old static attributes
        enhanced_attrs.pop('mode', None)
        enhanced_attrs.pop('dspy_generated', None)
        
        return enhanced_attrs
    
    def _adaptive_fallback_attributes(self, entity_class: str, entity_text: str, domain_context: Dict) -> Dict:
        """Generate semantic attributes using adaptive rules"""
        
        domain = domain_context['domain_category']
        context = domain_context['semantic_context']
        
        # Adaptive role assignment based on learned domain
        if 'name' in entity_class.lower():
            if domain == 'healthcare':
                role = 'healthcare_subject'
            elif domain == 'legal':
                role = 'legal_entity'
            elif domain == 'financial':
                role = 'business_entity'
            else:
                role = 'person_or_organization'
        elif any(term in entity_class.lower() for term in ['amount', 'value', 'cost', 'price']):
            role = 'monetary_figure'
        elif any(term in entity_class.lower() for term in ['date', 'time', 'deadline']):
            role = 'temporal_reference'
        elif 'medication' in entity_class.lower():
            role = 'therapeutic_agent'
        elif 'contract' in entity_class.lower():
            role = 'legal_instrument'
        else:
            role = f"{domain}_entity"
        
        # Adaptive category based on content
        if entity_text:
            if any(char.isdigit() for char in entity_text):
                if '$' in entity_text or '€' in entity_text:
                    category = 'monetary_amount'
                elif 'mg' in entity_text or 'ml' in entity_text:
                    category = 'medical_measurement'
                else:
                    category = 'quantitative_data'
            else:
                category = 'categorical_identifier'
        else:
            category = 'missing_data'
        
        # Adaptive significance
        if context == 'contractual_relationships':
            significance = 'contractual_obligation'
        elif context == 'patient_care':
            significance = 'clinical_information'
        elif context == 'financial_transactions':
            significance = 'financial_data'
        else:
            significance = 'extracted_information'
        
        return {
            'functional_role': role,
            'semantic_category': category,
            'contextual_significance': significance,
            'inferred_domain': domain,
            'semantic_context': context,
            'inference_confidence': domain_context.get('confidence_score', 1)
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


async def enhance_extraction_metadata_adaptive(extractions: List[Dict], document_schema: Dict = None) -> List[Dict]:
    """Main function for adaptive metadata enhancement (async)"""
    print(f"🧠 [ADAPTIVE] Starting async enhancement for {len(extractions)} extractions")
    enhancer = AdaptiveMetadataEnhancer()
    enhanced = await enhancer.enhance_extractions(extractions, document_schema)
    print(f"🧠 [ADAPTIVE] Enhanced {len(enhanced)} extractions")
    if enhanced:
        sample_attrs = enhanced[0].get('attributes', {})
        print(f"🧠 [ADAPTIVE] Sample enhanced attributes: {list(sample_attrs.keys())}")
    return enhanced