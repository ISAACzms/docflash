"""
Reinforcement Learning Feedback Analyzer
Analyzes user feedback and optimizes prompts for better example generation
"""

import json
import os
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import uuid


class FeedbackAnalyzer:
    """Analyzes user feedback and optimizes prompt generation"""
    
    def __init__(self, storage_path: str = "feedback_storage"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        # In-memory feedback store (will be persisted to files)
        self.feedback_store: Dict[str, Any] = {}
        
        # Prompt optimization strategies
        self.optimization_strategies = {
            'low_rating': self._handle_low_rating_feedback,
            'wrong_extraction': self._handle_wrong_extraction_feedback,
            'missing_fields': self._handle_missing_fields_feedback,
            'format_issues': self._handle_format_issues_feedback
        }
    
    def store_feedback(self, session_id: str, example_index: int, 
                      feedback_type: str, example_data: Dict, 
                      detailed_feedback: Optional[Dict] = None) -> bool:
        """Store user feedback for analysis"""
        try:
            feedback_id = f"{session_id}_{example_index}"
            
            self.feedback_store[feedback_id] = {
                'session_id': session_id,
                'example_index': example_index,
                'feedback_type': feedback_type,
                'example_data': example_data,
                'detailed_feedback': detailed_feedback,
                'timestamp': datetime.now().isoformat(),
                'processed': False
            }
            
            # Persist to file
            self._persist_feedback(feedback_id)
            return True
            
        except Exception as e:
            print(f"❌ [RL] Error storing feedback: {e}")
            return False
    
    def analyze_feedback_for_document_class(self, document_class: str) -> Dict[str, Any]:
        """Analyze all feedback for a specific document class"""
        
        try:
            # Find all feedback for this document class
            relevant_feedback = []
            for feedback_id, feedback in self.feedback_store.items():
                try:
                    if feedback and isinstance(feedback, dict):
                        example_data = feedback.get('example_data')
                        if example_data and isinstance(example_data, dict):
                            if example_data.get('document_class') == document_class:
                                relevant_feedback.append(feedback)
                except Exception as feedback_error:
                    print(f"⚠️ [RL] Skipping malformed feedback entry {feedback_id}: {feedback_error}")
                    continue
            
            if not relevant_feedback:
                return {'status': 'no_feedback', 'suggestions': []}
            
            # Analyze feedback patterns
            analysis = {
                'total_feedback': len(relevant_feedback),
                'positive_feedback': len([f for f in relevant_feedback if f['feedback_type'] == 'positive']),
                'negative_feedback': len([f for f in relevant_feedback if f['feedback_type'] == 'negative']),
                'detailed_feedback': len([f for f in relevant_feedback if f.get('detailed_feedback')]),
                'common_issues': self._identify_common_issues(relevant_feedback),
                'rating_distribution': self._analyze_ratings(relevant_feedback),
                'optimization_suggestions': self._generate_optimization_suggestions(relevant_feedback),
                'prompt_improvements': self._suggest_prompt_improvements(relevant_feedback)
            }
            
            return analysis
        
        except Exception as e:
            print(f"❌ [RL] Error analyzing feedback for document class {document_class}: {e}")
            return {'status': 'error', 'suggestions': []}
    
    def optimize_prompt_for_feedback(self, original_prompt: str, feedback_analysis: Dict) -> str:
        """Generate an improved prompt based on feedback analysis"""
        
        if feedback_analysis.get('status') == 'no_feedback':
            return original_prompt
        
        optimizations = []
        
        # Apply optimization strategies based on common issues
        for issue, count in feedback_analysis.get('common_issues', {}).items():
            if count > 0 and issue in self.optimization_strategies:
                optimization = self.optimization_strategies[issue](original_prompt, count)
                if optimization:
                    optimizations.append(optimization)
        
        # If we have specific prompt improvements from analysis
        prompt_improvements = feedback_analysis.get('prompt_improvements', [])
        optimizations.extend(prompt_improvements)
        
        if not optimizations:
            return original_prompt
        
        # Apply optimizations to the prompt
        optimized_prompt = self._apply_optimizations(original_prompt, optimizations)
        
        print(f"🧠 [RL] Applied {len(optimizations)} optimizations to prompt")
        return optimized_prompt
    
    def get_feedback_stats(self, document_class: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        
        feedback_list = list(self.feedback_store.values())
        
        if document_class:
            feedback_list = [
                f for f in feedback_list 
                if f['example_data'].get('document_class') == document_class
            ]
        
        if not feedback_list:
            return {'total': 0, 'stats': {}}
        
        positive_count = len([f for f in feedback_list if f['feedback_type'] == 'positive'])
        negative_count = len([f for f in feedback_list if f['feedback_type'] == 'negative'])
        
        # Calculate average rating from detailed feedback (only numeric ratings)
        numeric_ratings = []
        for feedback in feedback_list:
            detailed = feedback.get('detailed_feedback', {})
            rating = detailed.get('rating') if detailed else None
            # Only include numeric ratings in average calculation
            if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                numeric_ratings.append(rating)
        
        avg_rating = statistics.mean(numeric_ratings) if numeric_ratings else None
        
        return {
            'total': len(feedback_list),
            'positive': positive_count,
            'negative': negative_count,
            'satisfaction_rate': positive_count / len(feedback_list) if feedback_list else 0,
            'average_rating': avg_rating,
            'total_ratings': len(numeric_ratings),
            'document_class': document_class
        }
    
    def _identify_common_issues(self, feedback_list: List[Dict]) -> Dict[str, int]:
        """Identify common issues from detailed feedback"""
        
        issues = {
            'wrong_extraction': 0,
            'incorrect_text': 0,
            'missing_fields': 0,
            'format_issues': 0,
            'low_rating': 0
        }
        
        for feedback in feedback_list:
            try:
                detailed = feedback.get('detailed_feedback', {}) if feedback else {}
                if not isinstance(detailed, dict):
                    detailed = {}
                
                # Count specific issues
                for issue in detailed.get('issues', []):
                    if issue and issue.replace('-', '_') in issues:
                        issues[issue.replace('-', '_')] += 1
                
                # Count low ratings (including 0 ratings)
                rating = detailed.get('rating', 5)
                if isinstance(rating, (int, float)) and rating <= 2:
                    issues['low_rating'] += 1
                    
                # Analyze free-text comments for additional insights
                comments = detailed.get('comments', '').lower() if detailed.get('comments') else ''
                if comments:
                    # Look for common improvement themes in comments
                    if any(word in comments for word in ['currency', 'unit', 'format', 'type', 'specify']):
                        issues['format_issues'] += 1
                    if any(word in comments for word in ['missing', 'add', 'include', 'need']):
                        issues['missing_fields'] += 1
                    if any(word in comments for word in ['wrong', 'incorrect', 'error', 'fix']):
                        issues['wrong_extraction'] += 1
            except Exception as issue_error:
                print(f"⚠️ [RL] Error processing feedback item: {issue_error}")
                continue
        
        return issues
    
    def _analyze_ratings(self, feedback_list: List[Dict]) -> Dict[str, int]:
        """Analyze rating distribution (handles both numeric 1-5 and string positive/negative ratings)"""
        
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'positive': 0, 'negative': 0}
        
        for feedback in feedback_list:
            try:
                detailed = feedback.get('detailed_feedback', {}) if feedback else {}
                if isinstance(detailed, dict):
                    rating = detailed.get('rating')
                    
                    # Handle numeric ratings (1-5 stars)
                    if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                        distribution[int(rating)] += 1
                    # Handle string ratings (positive/negative for prompt feedback)
                    elif isinstance(rating, str) and rating in ['positive', 'negative']:
                        distribution[rating] += 1
                        
            except Exception as rating_error:
                print(f"⚠️ [RL] Error processing rating: {rating_error}")
                continue
        
        return distribution
    
    def _generate_optimization_suggestions(self, feedback_list: List[Dict]) -> List[str]:
        """Generate high-level optimization suggestions"""
        
        suggestions = []
        
        # Check feedback patterns
        negative_ratio = len([f for f in feedback_list if f['feedback_type'] == 'negative']) / len(feedback_list)
        
        if negative_ratio > 0.3:
            suggestions.append("High negative feedback rate - consider improving example quality")
        
        # Check rating patterns (handle both numeric and string ratings)
        low_ratings = []
        for f in feedback_list:
            rating = f.get('detailed_feedback', {}).get('rating', 5)
            # Count numeric ratings <= 2 or string 'negative' as low ratings
            if (isinstance(rating, (int, float)) and rating <= 2) or rating == 'negative':
                low_ratings.append(f)
        
        if len(low_ratings) > len(feedback_list) * 0.2:
            suggestions.append("Low ratings detected - review extraction accuracy and completeness")
        
        return suggestions
    
    def _suggest_prompt_improvements(self, feedback_list: List[Dict]) -> List[str]:
        """Suggest specific prompt improvements based on feedback"""
        
        improvements = []
        
        # Analyze common issues
        common_issues = self._identify_common_issues(feedback_list)
        
        if common_issues.get('wrong_extraction', 0) > 0:
            improvements.append(
                "Add more explicit field mapping instructions: 'Ensure each extraction maps to the correct attribute name as specified in the schema.'"
            )
        
        if common_issues.get('missing_fields', 0) > 0:
            improvements.append(
                "Add completeness instruction: 'Review the entire document to ensure all required fields from the schema are identified and extracted.'"
            )
        
        if common_issues.get('format_issues', 0) > 0:
            improvements.append(
                "Add format validation instruction: 'Ensure all extractions follow the exact JSON schema format with proper data types and structure.'"
            )
            
        # Extract specific improvements from user comments
        for feedback in feedback_list:
            comments = feedback.get('detailed_feedback', {}).get('comments', '')
            if comments:
                # Look for specific feedback patterns and convert to prompt improvements
                if 'currency' in comments.lower() or 'usd' in comments.lower():
                    improvements.append(
                        f"Currency formatting instruction: 'When extracting monetary values, include currency notation (e.g., $25,000 USD). User feedback: {comments[:100]}...'"
                    )
                if 'type' in comments.lower() and ('add' in comments.lower() or 'include' in comments.lower()):
                    improvements.append(
                        f"Enhanced detail instruction: 'Provide more specific detail and context in extractions. User feedback: {comments[:100]}...'"
                    )
        
        return improvements
    
    def _handle_low_rating_feedback(self, prompt: str, count: int) -> Optional[str]:
        """Handle optimization for low rating feedback"""
        if count > 1:
            return "Add quality emphasis: 'Focus on accuracy and completeness. Double-check each extraction against the source text.'"
        return None
    
    def _handle_wrong_extraction_feedback(self, prompt: str, count: int) -> Optional[str]:
        """Handle optimization for wrong extraction feedback"""
        if count > 1:
            return "Add mapping verification: 'Verify that each extracted value matches the correct attribute from the schema definition.'"
        return None
    
    def _handle_missing_fields_feedback(self, prompt: str, count: int) -> Optional[str]:
        """Handle optimization for missing fields feedback"""
        if count > 1:
            return "Add completeness check: 'Ensure all schema attributes are addressed. If a field cannot be found, note it explicitly.'"
        return None
    
    def _handle_format_issues_feedback(self, prompt: str, count: int) -> Optional[str]:
        """Handle optimization for format issues feedback"""
        if count > 1:
            return "Add format validation: 'Follow the exact JSON structure provided in the schema. Validate data types and field names.'"
        return None
    
    def _apply_optimizations(self, original_prompt: str, optimizations: List[str]) -> str:
        """Apply optimizations to create an improved prompt"""
        
        # Insert optimizations into the prompt
        optimization_text = "\n\nIMPORTANT QUALITY IMPROVEMENTS based on user feedback:\n"
        for i, opt in enumerate(optimizations, 1):
            optimization_text += f"{i}. {opt}\n"
        
        # Insert before the final instruction or at the end
        if "Please generate" in original_prompt:
            parts = original_prompt.split("Please generate", 1)
            return parts[0] + optimization_text + "\nPlease generate" + parts[1]
        else:
            return original_prompt + optimization_text
    
    def _persist_feedback(self, feedback_id: str) -> None:
        """Persist feedback to file"""
        try:
            feedback_file = os.path.join(self.storage_path, f"{feedback_id}.json")
            with open(feedback_file, 'w') as f:
                json.dump(self.feedback_store[feedback_id], f, indent=2)
        except Exception as e:
            print(f"❌ [RL] Error persisting feedback: {e}")
    
    def load_stored_feedback(self) -> None:
        """Load previously stored feedback from files"""
        try:
            if not os.path.exists(self.storage_path):
                return
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_path, filename)
                    with open(file_path, 'r') as f:
                        feedback_data = json.load(f)
                        feedback_id = filename[:-5]  # Remove .json extension
                        self.feedback_store[feedback_id] = feedback_data
            
            print(f"🧠 [RL] Loaded {len(self.feedback_store)} feedback entries")
            
        except Exception as e:
            print(f"❌ [RL] Error loading stored feedback: {e}")
    
    def delete_feedback_for_document_class(self, document_class: str) -> bool:
        """Delete all feedback data for a specific document class"""
        try:
            deleted_count = 0
            feedback_ids_to_delete = []
            
            # Find all feedback entries for this document class
            for feedback_id, feedback_data in self.feedback_store.items():
                if feedback_data.get('example_data', {}).get('document_class') == document_class:
                    feedback_ids_to_delete.append(feedback_id)
            
            # Delete from memory and files
            for feedback_id in feedback_ids_to_delete:
                # Delete from memory
                if feedback_id in self.feedback_store:
                    del self.feedback_store[feedback_id]
                    deleted_count += 1
                
                # Delete file if it exists
                feedback_file = os.path.join(self.storage_path, f"{feedback_id}.json")
                if os.path.exists(feedback_file):
                    os.remove(feedback_file)
            
            if deleted_count > 0:
                print(f"🧠 [RL] Deleted {deleted_count} feedback entries for document class: {document_class}")
            else:
                print(f"🧠 [RL] No feedback entries found for document class: {document_class}")
            
            return deleted_count > 0
            
        except Exception as e:
            print(f"❌ [RL] Error deleting feedback for document class {document_class}: {e}")
            return False


# Global feedback analyzer instance
feedback_analyzer = FeedbackAnalyzer()

# Load existing feedback on startup
feedback_analyzer.load_stored_feedback()