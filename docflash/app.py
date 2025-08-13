import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import aiofiles
import langextract as lx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langextract.data import ExampleData, Extraction

from .config import config
from .database import db

# Import existing modules
from .ocr_pipeline import PDFOCRPipeline
from .providers import ModelFactory

# Import RL feedback analyzer
from .rl_feedback_analyzer import feedback_analyzer

# Import DSPy integration 
from .dspy_integration import get_dspy_pipeline, should_trigger_optimization

# DSPy configuration
DSPY_ENABLED = True  # Feature flag to enable/disable DSPy

# FastAPI app initialization
app = FastAPI(
    title="Doc Flash - Document Intelligence Platform",
    description="Fast, intelligent document processing with the power of AI",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Configuration
app.state.upload_folder = "uploads"
app.state.output_folder = "outputs"
app.state.max_content_length = 16 * 1024 * 1024  # 16MB

# Create directories
os.makedirs(app.state.upload_folder, exist_ok=True)
os.makedirs(app.state.output_folder, exist_ok=True)

# Static files and templates
app.mount("/static", StaticFiles(directory="docflash/static"), name="static")
templates = Jinja2Templates(directory="docflash/templates")

# Global state for tracking tasks
active_tasks: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}


def get_all_document_classes() -> List[str]:
    """Get all unique document classes from templates, feedback, and recent usage"""
    document_classes = set()
    
    try:
        # Get from registered templates
        templates = db.list_templates()
        for template in templates:
            document_classes.add(template.get('document_class', ''))
            
        # Get from feedback store
        feedback_analyzer.load_stored_feedback()
        for feedback in feedback_analyzer.feedback_store.values():
            example_data = feedback.get('example_data', {})
            doc_class = example_data.get('document_class', '')
            if doc_class and doc_class.lower() != 'unknown':
                document_classes.add(doc_class)
                
        # Remove empty strings and 'unknown'
        document_classes = {cls for cls in document_classes if cls and cls.lower() != 'unknown'}
        
        return sorted(list(document_classes))
        
    except Exception as e:
        print(f"❌ Error getting document classes: {e}")
        return []


def create_language_model(temperature: float = 0.0):
    """Create language model instance based on environment configuration"""
    try:
        # Always use environment-configured model
        llm_config = config.llm_config
        llm_config.temperature = temperature

        return ModelFactory.create_model(llm_config)
    except Exception as e:
        print(f"Error creating language model: {e}")
        raise


# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/configure", response_class=HTMLResponse)
async def configure(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyze", response_class=HTMLResponse)
async def analyze(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request})


@app.get("/api/models")
async def get_available_models():
    """Get available models for current provider"""
    try:
        models = config.get_available_models()
        return {"success": True, "provider": config.provider, "models": models}
    except Exception as e:
        return {"success": False, "error": str(e), "models": {"main": [], "ocr": []}}


# WebSocket endpoint for real-time progress updates
@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    websocket_connections[task_id] = websocket

    try:
        while True:
            # Keep connection alive and wait for messages
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for task {task_id}")
    finally:
        if task_id in websocket_connections:
            del websocket_connections[task_id]


async def send_progress_update(
    task_id: str, message: str, progress: Optional[float] = None
):
    """Send progress update via WebSocket if connected"""
    if task_id in websocket_connections:
        try:
            update = {
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat(),
            }
            await websocket_connections[task_id].send_text(json.dumps(update))
        except Exception as e:
            print(f"Failed to send progress update: {e}")


# Background task for OCR processing
async def process_pdf_background(file_path: str, task_id: str):
    """Background task to process PDF with OCR"""
    try:
        active_tasks[task_id] = {
            "status": "processing",
            "start_time": datetime.now(),
            "progress": 0,
        }

        # Progress callback for real-time updates
        async def progress_callback(message: str):
            await send_progress_update(task_id, message)
            # Extract progress percentage if possible
            if "Completed" in message and "/" in message:
                try:
                    parts = message.split("Completed ")[1].split("/")
                    current = int(parts[0])
                    total = int(parts[1].split(" ")[0])
                    progress = (current / total) * 100
                    active_tasks[task_id]["progress"] = progress
                except:
                    pass

        # Initialize OCR pipeline
        pipeline = PDFOCRPipeline(max_concurrent=10)

        # Process the PDF
        result = await pipeline.process_pdf_to_markdown(file_path, progress_callback)

        # Store result
        active_tasks[task_id].update(
            {
                "status": "completed",
                "result": result,
                "end_time": datetime.now(),
                "progress": 100,
            }
        )

        await send_progress_update(task_id, "OCR processing completed!", 100)

    except Exception as e:
        active_tasks[task_id].update(
            {"status": "failed", "error": str(e), "end_time": datetime.now()}
        )
        await send_progress_update(task_id, f"Error: {str(e)}")


@app.post("/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks, pdf_file: UploadFile = File(...)
):
    """Upload PDF and start OCR processing in background"""

    # Validate file
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    if pdf_file.size > app.state.max_content_length:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Save uploaded file
        filename = f"{task_id}_{pdf_file.filename}"
        file_path = os.path.join(app.state.upload_folder, filename)

        async with aiofiles.open(file_path, "wb") as f:
            content = await pdf_file.read()
            await f.write(content)

        # Start background processing
        background_tasks.add_task(process_pdf_background, file_path, task_id)

        return {
            "success": True,
            "task_id": task_id,
            "message": "PDF uploaded successfully. Processing started.",
            "filename": pdf_file.filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = active_tasks[task_id]

    # Clean up completed tasks older than 1 hour
    if task["status"] in ["completed", "failed"]:
        if "end_time" in task:
            time_diff = datetime.now() - task["end_time"]
            if time_diff.total_seconds() > 3600:  # 1 hour
                del active_tasks[task_id]
                raise HTTPException(status_code=404, detail="Task expired")

    return task


# Template management endpoints
@app.post("/register_template")
async def register_template(request: Request):
    """Register a successful template configuration"""
    try:
        data = await request.json()
        document_class = data.get("document_class", "").strip()
        extraction_schema = data.get("extraction_schema", [])
        prompt_description = data.get("prompt_description", "")
        examples = data.get("examples", [])
        output_schema = data.get("output_schema", {})
        domain_name = data.get("domain_name")
        # Model is now configured via environment variables

        if not document_class:
            raise HTTPException(status_code=400, detail="Document class is required")

        if not domain_name:
            raise HTTPException(status_code=400, detail="Domain is required")

        if db.template_exists(document_class):
            raise HTTPException(
                status_code=409,
                detail=f'A template for "{document_class}" already exists. Use a different name or delete the existing one.',
            )

        template_id = db.register_template(
            document_class=document_class,
            extraction_schema=extraction_schema,
            prompt_description=prompt_description,
            examples=examples,
            output_schema=output_schema,
            domain_name=domain_name,
        )
        
        # Update domain statistics if domain was assigned
        if domain_name:
            db.update_domain_statistics()

        # Save DSPy optimization if available
        try:
            from .dspy_integration import get_dspy_pipeline
            dspy_pipeline = get_dspy_pipeline()
            if dspy_pipeline and dspy_pipeline.is_compiled:
                success = dspy_pipeline.save_optimized_model(document_class, domain_name)
                if success:
                    print(f"💾 [Domain Learning] Saved DSPy optimization for {document_class} in domain {domain_name}")
                else:
                    print(f"⚠️ [Domain Learning] Failed to save DSPy optimization for {document_class}")
        except Exception as e:
            print(f"⚠️ [Domain Learning] Error saving DSPy optimization: {e}")

        return {
            "success": True,
            "template_id": template_id,
            "message": f'Template "{document_class}" registered successfully!',
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to register template: {str(e)}"
        )


@app.get("/list_templates")
async def list_templates():
    """List all registered templates"""
    try:
        templates = db.list_templates()
        return {"success": True, "templates": templates}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list templates: {str(e)}"
        )


@app.get("/get_template/{document_class}")
async def get_template(document_class: str):
    """Get a specific template by document class"""
    try:
        template = db.get_template(document_class)

        if not template:
            raise HTTPException(
                status_code=404,
                detail=f'No template found for document class "{document_class}"',
            )

        return {"success": True, "template": template}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve template: {str(e)}"
        )


@app.get("/api/template/{document_class}")
async def get_template_api(document_class: str):
    """API endpoint to get a specific template by document class"""
    try:
        template = db.get_template(document_class)

        if not template:
            raise HTTPException(
                status_code=404,
                detail=f'No template found for document class "{document_class}"',
            )

        return {"success": True, "template": template}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve template: {str(e)}"
        )


@app.post("/analyze_document")
async def analyze_document(request: Request):
    """Analyze a document using a registered template"""
    try:
        data = await request.json()
        document_class = data.get("document_class", "")
        input_text = data.get("input_text", "")

        if not document_class or not input_text:
            raise HTTPException(
                status_code=400, detail="Document class and input text are required"
            )

        # Get the registered template
        print(f"🔧 [DEBUG] Getting template for: {document_class}")
        template = db.get_template(document_class)
        if not template:
            print(f"❌ [ERROR] Template not found: {document_class}")
            raise HTTPException(
                status_code=404,
                detail=f'No template found for document class "{document_class}"',
            )
        print(f"✅ [DEBUG] Template found: {document_class}")

        # Update usage statistics
        db.update_template_usage(document_class)

        # Use the template to run extraction
        print(f"🔍 Analyzing document using template: {document_class}")

        # Convert examples data to LangExtract format
        examples = []
        for ex_data in template["examples"]:
            extractions = []
            for ext_data in ex_data.get("extractions", []):
                extraction = Extraction(
                    extraction_class=ext_data.get("extraction_class", ""),
                    extraction_text=ext_data.get("extraction_text", ""),
                    attributes=ext_data.get("attributes", {}),
                )
                extractions.append(extraction)

            example = ExampleData(text=ex_data.get("text", ""), extractions=extractions)
            examples.append(example)

        # Determine optimal temperature based on schema modes (with fallback for legacy data)
        schema_attrs = template.get("extraction_schema", [])
        has_generate_fields = any(
            attr.get("mode", "extract") == "generate" for attr in schema_attrs
        )
        optimal_temperature = 0.3 if has_generate_fields else 0.0

        print(
            f"🔧 Template schema modes: {[(attr.get('attribute', 'unknown'), attr.get('mode', 'extract')) for attr in schema_attrs[:3]]}"
        )
        print(
            f"🌡️ Using temperature: {optimal_temperature} (has_generate_fields: {has_generate_fields})"
        )

        # Create extraction model instance with adaptive temperature
        print(f"🔧 [DEBUG] Creating language model with temperature: {optimal_temperature}")
        try:
            extraction_model = create_language_model(
                temperature=optimal_temperature
            )
            print(f"✅ [DEBUG] Language model created successfully: {type(extraction_model)}")
            print(f"🔧 [DEBUG] Language model attributes: {[attr for attr in dir(extraction_model) if not attr.startswith('_')]}")
            # Check for API key
            if hasattr(extraction_model, 'api_key'):
                print(f"🔧 [DEBUG] Model has api_key: {'***' if extraction_model.api_key else 'None'}")
            if hasattr(extraction_model, 'azure_ad_token_provider'):
                print(f"🔧 [DEBUG] Model has azure_ad_token_provider: {extraction_model.azure_ad_token_provider is not None}")
        except Exception as model_error:
            print(f"❌ [ERROR] Language model creation failed: {model_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create language model: {str(model_error)}"
            )

        # Configure extraction parameters
        extraction_params = {
            "text_or_documents": input_text,
            "prompt_description": template["prompt_description"],
            "examples": examples,
            "language_model_type": type(extraction_model),
            "language_model_params": {
                "model_id": extraction_model.model_id,
                "deployment_name": extraction_model.deployment_name,
                "azure_endpoint": extraction_model.azure_endpoint,
                "api_version": extraction_model.api_version,
                "azure_ad_token_provider": extraction_model.azure_ad_token_provider,
                "api_key": getattr(extraction_model, 'api_key', None),
                "temperature": extraction_model.temperature,
            },
            "extraction_passes": 1,
            "max_workers": 10,
            "max_char_buffer": 1000,
        }

        # Add provider-specific parameters (Gemini doesn't need these)
        if (
            not hasattr(extraction_model, "__class__")
            or "Gemini" not in extraction_model.__class__.__name__
        ):
            extraction_params["use_schema_constraints"] = False
            extraction_params["fence_output"] = True

        print(f"🔄 Running extraction for {document_class}...")
        try:
            print(f"🔧 [DEBUG] Extraction params: {list(extraction_params.keys())}")
            print(f"🔧 [DEBUG] Language model type: {type(extraction_model)}")
            print(f"🔧 [DEBUG] Examples count: {len(examples)}")
            print(f"🔧 [DEBUG] Language model params keys: {list(extraction_params['language_model_params'].keys())}")
            print(f"🔧 [DEBUG] API key present: {bool(extraction_params['language_model_params'].get('api_key'))}")
            print(f"🔧 [DEBUG] Token provider present: {bool(extraction_params['language_model_params'].get('azure_ad_token_provider'))}")
            result = lx.extract(**extraction_params)
            print(f"✅ [DEBUG] Extraction completed successfully")
        except Exception as extraction_error:
            print(f"❌ [ERROR] Extraction failed: {extraction_error}")
            print(f"🔧 [DEBUG] Full extraction error: {str(extraction_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Extraction failed: {str(extraction_error)}"
            )

        # Prepare extraction results
        extractions_list = []
        for extraction in result.extractions:
            extractions_list.append(
                {
                    "extraction_class": extraction.extraction_class,
                    "extraction_text": extraction.extraction_text,
                    "attributes": extraction.attributes or {},
                    "char_position": (
                        f"{extraction.char_interval.start_pos}-{extraction.char_interval.end_pos}"
                        if extraction.char_interval
                        else "unknown"
                    ),
                }
            )

        # Generate unique session ID for file management
        session_id = str(uuid.uuid4())[:8]

        # Save langextract results
        output_file = f"analysis_{session_id}.jsonl"
        lx.io.save_annotated_documents(
            [result], output_name=output_file, output_dir=app.state.output_folder
        )

        # Transform to target schema if provided
        final_output = None
        if template.get("output_schema"):
            print("🤖 Transforming to target schema...")
            transformation_prompt = f"""
Transform the following LangExtract extraction results into the specified output schema format.

**EXTRACTION RESULTS:**
{json.dumps(extractions_list, indent=2)}

**TARGET SCHEMA:**
{json.dumps(template['output_schema'], indent=2)}

**INSTRUCTIONS:**
1. Map each extraction to the most appropriate field in the target schema
2. Combine related extractions intelligently
3. Use page numbers from attributes when available - IMPORTANT: If the target schema expects page numbers as a list, ensure the list contains only unique values (no duplicates)
4. Fill as many schema fields as possible
5. Leave unmapped fields with their placeholder values

**PAGE NUMBER HANDLING:**
- If page_number or page_numbers field exists in target schema: extract unique page numbers only
- Convert single page numbers to integers, lists to arrays of unique integers
- If no page numbers are available in the extractions, use None for the field
- Example: if extractions show pages [1,1,2,1,3], output should be [1,2,3]
- Example: if no page info found, output should be None

**OUTPUT:** Return ONLY the filled JSON in the exact target schema format.
"""

            try:
                batch_result = list(extraction_model.infer([transformation_prompt]))
                transformation_result = batch_result[0][0]
                json_text = transformation_result.output

                # Clean response
                json_text = json_text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                if json_text.startswith("```"):
                    json_text = json_text[3:]

                final_output = json.loads(json_text)
            except Exception as e:
                print(f"Schema transformation failed: {e}")
                final_output = template["output_schema"]

        # Generate visualization
        output_file_path = os.path.join(app.state.output_folder, output_file)
        html_content = lx.visualize(output_file_path)
        if hasattr(html_content, "data"):
            visualization_html = html_content.data
        else:
            visualization_html = str(html_content)

        # Save visualization
        viz_file = f"analysis_viz_{session_id}.html"
        viz_path = os.path.join(app.state.output_folder, viz_file)
        async with aiofiles.open(viz_path, "w") as f:
            await f.write(visualization_html)

        # Save structured JSON output if it exists
        schema_file = f"analysis_schema_{session_id}.json"
        if final_output:
            schema_path = os.path.join(app.state.output_folder, schema_file)
            async with aiofiles.open(schema_path, "w") as f:
                await f.write(json.dumps(final_output, indent=2))

        return {
            "success": True,
            "document_class": document_class,
            "extractions": extractions_list,
            "final_output": final_output,
            "session_id": session_id,
            "visualization_file": viz_file,
            "schema_file": schema_file if final_output else None,
            "raw_file": output_file,
            "message": f'Document analyzed successfully using template "{document_class}"',
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.delete("/delete_template/{document_class}")
async def delete_template(document_class: str):
    """Delete a registered template and associated feedback data"""
    try:
        # Get template info before deletion to extract domain name
        template = db.get_template(document_class)
        domain_name = template.get("domain_name") if template else None
        
        success = db.delete_template(document_class)

        if success:
            # Clean up all feedback data for this document class
            try:
                from docflash.rl_feedback_analyzer import feedback_analyzer
                feedback_analyzer.delete_feedback_for_document_class(document_class)
                print(f"🧠 [RL] Cleaned up feedback data for document class: {document_class}")
            except Exception as e:
                print(f"⚠️ [RL] Warning: Could not clean up feedback data: {e}")
            
            # Clean up DSPy feedback if available
            try:
                dspy_pipeline = get_dspy_pipeline()
                if dspy_pipeline:
                    dspy_pipeline.delete_feedback_for_document_class(document_class)
                    print(f"🧠 [DSPy] Cleaned up DSPy feedback for document class: {document_class}")
            except Exception as e:
                print(f"⚠️ [DSPy] Warning: Could not clean up DSPy feedback: {e}")
            
            # Clean up DSPy saved model and metadata
            if domain_name:
                try:
                    dspy_pipeline = get_dspy_pipeline()
                    if dspy_pipeline:
                        cleanup_success = dspy_pipeline.delete_saved_model(document_class, domain_name)
                        if cleanup_success:
                            print(f"🗑️ [DSPy] Cleaned up saved model for {document_class} in domain {domain_name}")
                        else:
                            print(f"🔍 [DSPy] No saved model found for {document_class} in domain {domain_name}")
                except Exception as e:
                    print(f"⚠️ [DSPy] Warning: Could not clean up saved model: {e}")
            
            return {
                "success": True,
                "message": f'Template "{document_class}" and all associated data deleted successfully!',
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f'No template found for document class "{document_class}"',
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete template: {str(e)}"
        )


@app.get("/edit_template", response_class=HTMLResponse)
async def edit_template(request: Request, document_class: str = None):
    """Edit template page"""
    if not document_class:
        return templates.TemplateResponse("index.html", {"request": request})

    template = db.get_template(document_class)
    if not template:
        return templates.TemplateResponse("index.html", {"request": request})

    return templates.TemplateResponse(
        "edit_template.html", {"request": request, "template": template}
    )


@app.post("/update_template")
async def update_template(request: Request):
    """Update an existing template (document class name cannot be changed)"""
    try:
        data = await request.json()
        original_class = data.get("original_document_class", "").strip()
        document_class = data.get("document_class", "").strip()
        extraction_schema = data.get("extraction_schema", [])
        prompt_description = data.get("prompt_description", "")
        examples = data.get("examples", [])
        output_schema = data.get("output_schema", {})
        domain_name = data.get("domain_name")

        if not original_class or not document_class:
            raise HTTPException(status_code=400, detail="Document class is required")

        if not domain_name:
            raise HTTPException(status_code=400, detail="Domain is required")

        # Check if original template exists
        if not db.template_exists(original_class):
            raise HTTPException(
                status_code=404,
                detail=f'Original template "{original_class}" not found',
            )

        # Prevent document class name changes
        if original_class != document_class:
            raise HTTPException(
                status_code=400,
                detail=f'Document class name cannot be changed. Template name must remain "{original_class}". To change the name, delete this template and create a new one.'
            )

        # Update existing template (preserves metadata automatically)
        success = db.update_template(
            document_class=document_class,
            extraction_schema=extraction_schema,
            prompt_description=prompt_description,
            examples=examples,
            output_schema=output_schema,
            domain_name=domain_name,
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update template '{document_class}'"
            )

        # Save DSPy optimization after template update
        try:
            from .dspy_integration import get_dspy_pipeline
            dspy_pipeline = get_dspy_pipeline()
            if dspy_pipeline and dspy_pipeline.is_compiled:
                success = dspy_pipeline.save_optimized_model(document_class, domain_name)
                if success:
                    print(f"💾 [Domain Learning] Updated DSPy optimization for {document_class} in domain {domain_name}")
                else:
                    print(f"⚠️ [Domain Learning] Failed to update DSPy optimization for {document_class}")
        except Exception as e:
            print(f"⚠️ [Domain Learning] Error updating DSPy optimization: {e}")

        return {
            "success": True,
            "message": f'Template "{document_class}" updated successfully!',
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update template: {str(e)}"
        )


@app.post("/generate_examples")
async def generate_examples(request: Request):
    """Generate langextract examples and prompts from user-defined schema"""
    try:
        data = await request.json()
        sample_texts = data.get("sample_texts", [])
        extraction_schema = data.get("extraction_schema", [])
        output_schema = data.get("output_schema", {})
        # Model is now configured via environment variables
        document_class = data.get("document_class", "unknown")
        
        # RL feedback data for regeneration
        rl_session_id = data.get("rl_session_id")
        master_feedback = data.get("master_feedback", {})  # Current session feedback
        # Legacy support for old feedback format
        feedback_data = data.get("feedback_data", {})
        
        # CRITICAL FIX: Load ALL accumulated feedback for this document class (including prompt feedback)
        accumulated_master_feedback = {}
        current_session_feedback = {}
        if document_class and document_class.lower() != "unknown":
            try:
                from docflash.rl_feedback_analyzer import feedback_analyzer
                feedback_analyzer.load_stored_feedback()  # Ensure latest data is loaded
                
                # Find all master feedback entries AND current session feedback for this document class
                master_feedback_entries = []
                session_feedback_entries = []
                
                for feedback_id, feedback_data_entry in feedback_analyzer.feedback_store.items():
                    entry_doc_class = feedback_data_entry.get('example_data', {}).get('document_class')
                    feedback_source = feedback_data_entry.get('example_data', {}).get('feedback_source')
                    example_index = feedback_data_entry.get('example_index')
                    
                    if entry_doc_class == document_class:
                        if feedback_source == 'master':
                            master_feedback_entries.append(feedback_data_entry)
                        elif feedback_source == 'prompt_description':
                            # Include prompt feedback in current session data
                            session_feedback_entries.append(feedback_data_entry)
                            current_session_feedback['prompt'] = {
                                'type': feedback_data_entry.get('feedback_type', 'unknown'),
                                'detailed_feedback': feedback_data_entry.get('detailed_feedback'),
                                'example_data': feedback_data_entry.get('example_data', {})
                            }
                            print(f"🧠 [RL] Found prompt feedback for document class: {document_class}")
                
                if master_feedback_entries:
                    # Sort by timestamp, newest first
                    master_feedback_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                    # Pass ALL feedback entries to DSPy, but use most recent for display
                    accumulated_master_feedback = {
                        'primary': master_feedback_entries[0],  # Most recent for display
                        'all_entries': master_feedback_entries  # All entries for DSPy training
                    }
                    print(f"🧠 [RL] Loaded {len(master_feedback_entries)} accumulated master feedback entries for document class: {document_class}")
                    print(f"🔄 [RL] Using most recent master feedback: {master_feedback_entries[0].get('feedback_type', 'unknown')}")
                else:
                    print(f"🔍 [RL] No accumulated master feedback found for document class: {document_class}")
                    
            except Exception as e:
                print(f"⚠️ [RL] Error loading accumulated master feedback: {e}")
        
        # Use accumulated feedback if available, otherwise fall back to current session feedback
        effective_master_feedback = accumulated_master_feedback if accumulated_master_feedback else master_feedback
        
        # Merge current session feedback with any existing feedback_data
        if current_session_feedback:
            feedback_data = {**feedback_data, **current_session_feedback}
            print(f"🧠 [RL] Merged current session feedback: {list(current_session_feedback.keys())}")
        
        # CRITICAL FIX: Generate unique RL session ID if we have accumulated feedback but no current session
        if not rl_session_id and effective_master_feedback:
            rl_session_id = f"dspy_{document_class}_{int(time.time() * 1000)}"
            print(f"🔄 [RL] Generated RL session ID for DSPy with accumulated feedback: {rl_session_id}")
        
        if rl_session_id and effective_master_feedback:
            feedback_source = "accumulated" if accumulated_master_feedback else "current session"
            # Handle new structure where accumulated_master_feedback contains primary + all_entries
            display_feedback = effective_master_feedback
            if isinstance(effective_master_feedback, dict) and 'primary' in effective_master_feedback:
                display_feedback = effective_master_feedback['primary']
                
            print(f"🧠 [RL] Using {feedback_source} master feedback: {display_feedback.get('feedback_type', 'unknown')} for session {rl_session_id}")
            if display_feedback.get('detailed_feedback'):
                detailed = display_feedback['detailed_feedback']
                print(f"🧠 [RL] Detailed feedback: {detailed.get('rating', 0)}⭐, issues: {detailed.get('issues', [])}")
        elif rl_session_id and feedback_data:
            print(f"🧠 [RL] Regenerating with session feedback: {len(feedback_data)} items for session {rl_session_id}")
            if 'prompt' in feedback_data:
                print(f"🧠 [RL] Including prompt description feedback in regeneration")
        else:
            print(f"🔍 [RL] No feedback available for regeneration")
        
        # Validate document_class for duplicates (case-insensitive)
        if document_class and document_class.lower() != "unknown":
            existing_classes = get_all_document_classes()
            normalized_existing = [cls.lower().strip() for cls in existing_classes]
            if document_class.lower().strip() in normalized_existing:
                print(f"⚠️  [Validation] Document class '{document_class}' already exists")
                # Don't raise error, just log - user might be intentionally regenerating

        # 🧠 DSPy Integration: Try DSPy-optimized generation first (only if properly configured)
        dspy_available = False
        if DSPY_ENABLED:
            try:
                print(f"🧠 [DSPy] Checking DSPy availability for {document_class}")
                dspy_pipeline = get_dspy_pipeline()
                
                # Check if DSPy is actually configured and working
                import dspy
                if dspy.settings.lm is not None:
                    print("✅ [DSPy] DSPy is properly configured, attempting generation...")
                    dspy_result = dspy_pipeline.replace_generate_examples(
                        document_class=document_class,
                        sample_texts=sample_texts,
                        extraction_schema=extraction_schema,
                        output_schema=output_schema,
                        rl_session_id=rl_session_id,
                        master_feedback=effective_master_feedback,  # Use accumulated feedback
                        feedback_data=feedback_data  # Legacy support
                    )
                    
                    # Only return DSPy result if it was actually successful (not fallback)
                    if dspy_result.get('success', False) and not dspy_result.get('fallback_used', False):
                        # Check if optimization should be triggered
                        if should_trigger_optimization():
                            print("🔄 [DSPy] Triggering background optimization...")
                            asyncio.create_task(asyncio.to_thread(dspy_pipeline.optimize_from_feedback, document_class))
                        
                        print(f"✅ [DSPy] Successfully generated examples using DSPy pipeline")
                        return dspy_result
                    else:
                        print("⚠️  [DSPy] DSPy generation unsuccessful, falling back to current system")
                else:
                    print("⚠️  [DSPy] No LM configured, will use current DocFlash system")
                
            except Exception as e:
                print(f"⚠️  [DSPy] Failed to generate with DSPy: {e}")
                
        print("🔄 [System] Using current DocFlash example generation system")

        # Build schema description with mode awareness
        extract_fields = [
            item
            for item in extraction_schema
            if item.get("mode", "extract") == "extract"
        ]
        generate_fields = [
            item
            for item in extraction_schema
            if item.get("mode", "extract") == "generate"
        ]

        schema_description = ""
        if extract_fields:
            schema_description += "EXTRACT from document (exact text spans):\\n"
            schema_description += "\\n".join(
                [
                    f"- {item['attribute']}: {item['description']}"
                    for item in extract_fields
                ]
            )

        if generate_fields:
            if extract_fields:
                schema_description += "\\n\\n"
            schema_description += "GENERATE/INTERPRET (derived information):\\n"
            schema_description += "\\n".join(
                [
                    f"- {item['attribute']}: {item['description']}"
                    for item in generate_fields
                ]
            )

        # Combine multiple sample texts
        combined_sample_text = "\\n\\n--- SAMPLE SEPARATOR ---\\n\\n".join(sample_texts)

        # Mode-specific instructions
        mode_instructions = ""
        if generate_fields:
            mode_instructions = f"""

**IMPORTANT MODE INSTRUCTIONS:**
- EXTRACT fields should contain exact text spans from the document (use temperature 0.0-0.2)
- GENERATE fields should contain interpreted/derived information like summaries, analysis, or classifications (use temperature 0.3-0.7)
- For EXTRACT: Find and copy exact text from the document
- For GENERATE: Analyze, summarize, or interpret information to create new content"""

        # 🧠 RL OPTIMIZATION: Use feedback analysis to improve prompts
        # document_class already extracted above for DSPy integration
        
        # Reload stored feedback to ensure we have latest data (FastAPI restarts during dev)
        feedback_analyzer.load_stored_feedback()
        feedback_analysis = feedback_analyzer.analyze_feedback_for_document_class(document_class)
        
        print(f"🔍 [RL DEBUG] Looking for feedback for document_class: '{document_class}'")
        print(f"🔍 [RL DEBUG] Total feedback in store: {len(feedback_analyzer.feedback_store)}")
        print(f"🔍 [RL DEBUG] Analysis result: {feedback_analysis.get('status', 'unknown')}")
        
        rl_optimizations = ""
        if feedback_analysis.get('status') != 'no_feedback':
            print(f"🧠 [RL] Applying feedback-based optimizations for {document_class}")
            print(f"   - Found {feedback_analysis.get('total_feedback', 0)} feedback entries")
            print(f"   - Common issues: {list(feedback_analysis.get('common_issues', {}).keys())}")
            
            # Generate optimization instructions based on feedback
            optimizations = []
            common_issues = feedback_analysis.get('common_issues', {})
            
            if common_issues.get('wrong_extraction', 0) > 0:
                optimizations.append("- CRITICAL: Verify each extraction maps to correct schema attribute name")
            
            if common_issues.get('missing_fields', 0) > 0:
                optimizations.append("- COMPLETENESS: Ensure all schema fields are addressed in examples")
                
            if common_issues.get('format_issues', 0) > 0:
                optimizations.append("- FORMAT: Follow exact JSON structure with proper data types")
                
            if common_issues.get('low_rating', 0) > 0:
                optimizations.append("- QUALITY: Focus on accuracy and realistic example generation")
                
            # Add specific feedback from user comments
            prompt_improvements = feedback_analysis.get('prompt_improvements', [])
            for improvement in prompt_improvements:
                optimizations.append(f"- USER FEEDBACK: {improvement}")
            
            if optimizations:
                rl_optimizations = f"""

**🧠 RL FEEDBACK OPTIMIZATIONS (based on user feedback):**
{chr(10).join(optimizations)}
- These improvements come from analyzing user feedback on previous examples
"""
                print(f"✅ [RL] Applied {len(optimizations)} feedback-based optimizations")

        # Enhanced few-shot prompt with mode awareness and RL optimizations
        example_generation_prompt = f"""
You are an expert in information extraction using LangExtract with mode-aware processing capabilities. Your task is to analyze provided text(s) and schema to generate high-quality extraction examples.

**SAMPLE TEXT(S):**
{combined_sample_text}

**EXTRACTION SCHEMA:**
{schema_description}{mode_instructions}{rl_optimizations}

**EXAMPLE OF MODE-AWARE LANGEXTRACT FORMAT:**
Here's an example showing the difference between EXTRACT and GENERATE modes:

Sample Input: "Patient John Smith, age 45, was prescribed 400mg Ibuprofen twice daily for chronic back pain. This is his second prescription this year."
Schema: medication (EXTRACT), dosage (EXTRACT), risk_level (GENERATE), patient_summary (GENERATE)
Good Output:
{{
    "prompt_description": "Extract medication details exactly as written, and generate risk assessments and patient summaries based on the information.",
    "examples": [
        {{
            "text": "Patient John Smith, age 45, was prescribed 400mg Ibuprofen twice daily for chronic back pain. This is his second prescription this year.",
            "extractions": [
                {{
                    "extraction_class": "medication",
                    "extraction_text": "Ibuprofen",
                    "attributes": {{"section": "medications", "page_number": "1", "mode": "extract"}}
                }},
                {{
                    "extraction_class": "dosage", 
                    "extraction_text": "400mg twice daily",
                    "attributes": {{"section": "medications", "page_number": "1", "mode": "extract"}}
                }},
                {{
                    "extraction_class": "risk_level",
                    "extraction_text": "Moderate - frequent NSAID use for chronic condition",
                    "attributes": {{"section": "analysis", "page_number": "1", "mode": "generate"}}
                }},
                {{
                    "extraction_class": "patient_summary",
                    "extraction_text": "45-year-old with recurring chronic back pain, multiple prescriptions this year",
                    "attributes": {{"section": "analysis", "page_number": "1", "mode": "generate"}}
                }}
            ]
        }}
    ]
}}

**YOUR TASK:**
1. Create a clear prompt_description explaining what to extract vs generate from similar documents
2. Generate 2-4 diverse, realistic ExampleData instances 
3. For EXTRACT fields: Use exact text spans from the document - copy verbatim
4. For GENERATE fields: Create interpreted/derived content like summaries, risk assessments, classifications
5. Include meaningful attributes with "mode": "extract" or "mode": "generate"
6. Ensure extraction_class names match the provided schema attributes exactly
7. Make extractions realistic for the document type
8. Order extractions by appearance in text when possible

**OUTPUT FORMAT:**
Return ONLY a JSON object with this exact structure:
{{
    "prompt_description": "Extract [specific information] including [key elements] in order of appearance. Use exact text for extractions.",
    "examples": [
        {{
            "text": "Example text from or similar to the provided samples...",
            "extractions": [
                {{
                    "extraction_class": "schema_attribute_name",
                    "extraction_text": "exact text from document",
                    "attributes": {{"section": "category", "page_number": "1", "additional_context": "value"}}
                }}
            ]
        }}
    ]
}}
"""

        print(f"🤖 Generating examples and prompts using environment-configured model...")
        # Create model instance for example generation
        example_gen_model = create_language_model(temperature=0.1)
        batch_result = list(example_gen_model.infer([example_generation_prompt]))
        llm_response = batch_result[0][0]

        # Parse LLM response
        response_text = llm_response.output

        # Clean and parse JSON
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        if response_text.startswith("```"):
            response_text = response_text[3:]

        generated_data = json.loads(response_text)

        # Enhance examples with metadata for RL tracking
        enhanced_examples = generated_data.get("examples", [])
        for example in enhanced_examples:
            example["document_class"] = document_class
            example["generation_timestamp"] = datetime.now().isoformat()
        
        rl_session_id = str(uuid.uuid4())[:8]
        rl_message = "Examples generated successfully!"
        
        # Add system indicator
        if DSPY_ENABLED:
            import dspy
            if dspy.settings.lm is None:
                rl_message = "Examples generated successfully using current DocFlash system! (DSPy not configured)"
            else:
                rl_message = "Examples generated successfully using current DocFlash system! (DSPy fallback)"
        else:
            rl_message = "Examples generated successfully using current DocFlash system! (DSPy disabled)"
        
        # Add RL-specific messaging
        if feedback_analysis.get('status') != 'no_feedback':
            total_optimizations = len([opt for opt in [
                feedback_analysis.get('common_issues', {}).get('wrong_extraction', 0),
                feedback_analysis.get('common_issues', {}).get('missing_fields', 0),
                feedback_analysis.get('common_issues', {}).get('format_issues', 0),
                feedback_analysis.get('common_issues', {}).get('low_rating', 0)
            ] if opt > 1])
            
            if total_optimizations > 0:
                rl_message += f" 🧠 RL applied {total_optimizations} improvements from user feedback!"
        
        return {
            "success": True,
            "prompt_description": generated_data.get("prompt_description", ""),
            "examples": enhanced_examples,
            "message": rl_message,
            "session_id": rl_session_id,  # For RL feedback tracking
            "rl_enabled": True,
            "rl_feedback_applied": feedback_analysis.get('status') != 'no_feedback',
            "document_class": document_class
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate examples: {str(e)}"
        )


@app.post("/run_extraction")
async def run_extraction(request: Request):
    """Run the complete extraction pipeline"""
    try:
        data = await request.json()
        input_text = data.get("input_text", "")
        prompt_description = data.get("prompt_description", "")
        examples_data = data.get("examples", [])
        output_schema = data.get("output_schema", {})
        # Model is now configured via environment variables
        extraction_passes = data.get("extraction_passes", 1)
        max_workers = data.get("max_workers", 10)
        max_char_buffer = data.get("max_char_buffer", 1000)

        # Convert examples data to LangExtract format
        examples = []
        for ex_data in examples_data:
            extractions = []
            for ext_data in ex_data.get("extractions", []):
                extraction = Extraction(
                    extraction_class=ext_data.get("extraction_class", ""),
                    extraction_text=ext_data.get("extraction_text", ""),
                    attributes=ext_data.get("attributes", {}),
                )
                extractions.append(extraction)

            example = ExampleData(text=ex_data.get("text", ""), extractions=extractions)
            examples.append(example)

        print(f"🔍 Running LangExtract with environment-configured model...")

        # Determine optimal temperature based on extraction attributes in examples
        has_generate_fields = False
        extraction_classes = set()
        for ex_data in examples_data:
            for ext_data in ex_data.get("extractions", []):
                extraction_class = ext_data.get("extraction_class", "")
                mode = ext_data.get("attributes", {}).get("mode", "extract")
                extraction_classes.add((extraction_class, mode))
                if mode == "generate":
                    has_generate_fields = True

        optimal_temperature = 0.3 if has_generate_fields else 0.0

        print(f"🔧 Extraction classes detected: {list(extraction_classes)[:3]}")
        print(
            f"🌡️ Using temperature: {optimal_temperature} (has_generate_fields: {has_generate_fields})"
        )

        # Create extraction model instance with adaptive temperature
        extraction_model = create_language_model(
            temperature=optimal_temperature
        )

        # Configure extraction parameters - safely extract model params
        language_model_params = {
            "temperature": optimal_temperature,  # Use the calculated optimal temperature
        }
        
        # Safely extract model parameters based on what's available
        if hasattr(extraction_model, 'model_id'):
            language_model_params["model_id"] = extraction_model.model_id
        if hasattr(extraction_model, 'deployment_name'):
            language_model_params["deployment_name"] = extraction_model.deployment_name  
        if hasattr(extraction_model, 'azure_endpoint'):
            language_model_params["azure_endpoint"] = extraction_model.azure_endpoint
        if hasattr(extraction_model, 'api_version'):
            language_model_params["api_version"] = extraction_model.api_version
        if hasattr(extraction_model, 'azure_ad_token_provider'):
            language_model_params["azure_ad_token_provider"] = extraction_model.azure_ad_token_provider
        
        # Most importantly, include the API key if available
        if hasattr(extraction_model, 'api_key'):
            language_model_params["api_key"] = extraction_model.api_key
            
        print(f"🔧 [Model] Using environment-configured model with params: {list(language_model_params.keys())}")
        
        extraction_params = {
            "text_or_documents": input_text,
            "prompt_description": prompt_description,
            "examples": examples,
            "language_model_type": type(extraction_model),
            "language_model_params": language_model_params,
            "extraction_passes": extraction_passes,
            "max_workers": max_workers,
            "max_char_buffer": max_char_buffer,
        }

        # Add provider-specific parameters (Gemini doesn't need these)
        if (
            not hasattr(extraction_model, "__class__")
            or "Gemini" not in extraction_model.__class__.__name__
        ):
            extraction_params["use_schema_constraints"] = False
            extraction_params["fence_output"] = True

        # Run extraction
        result = lx.extract(**extraction_params)

        # Prepare extraction results
        extractions_list = []
        for extraction in result.extractions:
            extractions_list.append(
                {
                    "extraction_class": extraction.extraction_class,
                    "extraction_text": extraction.extraction_text,
                    "attributes": extraction.attributes or {},
                    "char_position": (
                        f"{extraction.char_interval.start_pos}-{extraction.char_interval.end_pos}"
                        if extraction.char_interval
                        else "unknown"
                    ),
                }
            )

        # Generate unique session ID
        session_id = str(uuid.uuid4())[:8]

        # Save results
        output_file = f"analysis_{session_id}.jsonl"
        lx.io.save_annotated_documents(
            [result], output_name=output_file, output_dir=app.state.output_folder
        )

        # Transform to target schema if provided
        final_output = None
        if output_schema:
            print("🤖 Transforming to target schema...")
            transformation_prompt = f"""
Transform the following LangExtract extraction results into the specified output schema format.

**EXTRACTION RESULTS:**
{json.dumps(extractions_list, indent=2)}

**TARGET SCHEMA:**
{json.dumps(output_schema, indent=2)}

**INSTRUCTIONS:**
1. Map each extraction to the most appropriate field in the target schema
2. Combine related extractions intelligently
3. Use page numbers from attributes when available - IMPORTANT: If the target schema expects page numbers as a list, ensure the list contains only unique values (no duplicates)
4. Fill as many schema fields as possible
5. Leave unmapped fields with their placeholder values

**PAGE NUMBER HANDLING:**
- If page_number or page_numbers field exists in target schema: extract unique page numbers only
- Convert single page numbers to integers, lists to arrays of unique integers
- If no page numbers are available in the extractions, use None for the field
- Example: if extractions show pages [1,1,2,1,3], output should be [1,2,3]
- Example: if no page info found, output should be None

**OUTPUT:** Return ONLY the filled JSON in the exact target schema format.
"""

            try:
                batch_result = list(extraction_model.infer([transformation_prompt]))
                transformation_result = batch_result[0][0]
                json_text = transformation_result.output

                # Clean response
                json_text = json_text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                if json_text.startswith("```"):
                    json_text = json_text[3:]

                final_output = json.loads(json_text)
            except Exception as e:
                print(f"Schema transformation failed: {e}")
                final_output = output_schema

        # Generate visualization
        output_file_path = os.path.join(app.state.output_folder, output_file)
        html_content = lx.visualize(output_file_path)
        if hasattr(html_content, "data"):
            visualization_html = html_content.data
        else:
            visualization_html = str(html_content)

        # Save visualization
        viz_file = f"analysis_viz_{session_id}.html"
        viz_path = os.path.join(app.state.output_folder, viz_file)
        async with aiofiles.open(viz_path, "w") as f:
            await f.write(visualization_html)

        # Save structured JSON output if it exists
        schema_file = f"analysis_schema_{session_id}.json"
        if final_output:
            schema_path = os.path.join(app.state.output_folder, schema_file)
            async with aiofiles.open(schema_path, "w") as f:
                await f.write(json.dumps(final_output, indent=2))

        return {
            "success": True,
            "extractions": extractions_list,
            "final_output": final_output,
            "session_id": session_id,
            "visualization_file": viz_file,
            "schema_file": schema_file if final_output else None,
            "raw_file": output_file,
            "message": f"Extraction completed successfully! Found {len(extractions_list)} entities.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# File serving endpoints
@app.get("/outputs/{filename}")
async def serve_output_file(filename: str):
    """Serve files from the outputs directory"""
    file_path = os.path.join(app.state.output_folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type
    if filename.endswith(".html"):
        return FileResponse(file_path, media_type="text/html")
    elif filename.endswith(".json"):
        return FileResponse(file_path, media_type="application/json")
    elif filename.endswith(".jsonl"):
        return FileResponse(file_path, media_type="application/json")
    else:
        return FileResponse(file_path)


@app.get("/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """Download generated files"""
    try:
        if file_type == "visualization":
            # Try both visualization and analysis naming patterns
            filename = f"visualization_{session_id}.html"
            if not os.path.exists(os.path.join(app.state.output_folder, filename)):
                filename = f"analysis_viz_{session_id}.html"
            media_type = "text/html"
        elif file_type == "extractions":
            # Try both extraction and analysis naming patterns
            filename = f"extraction_{session_id}.jsonl"
            if not os.path.exists(os.path.join(app.state.output_folder, filename)):
                filename = f"analysis_{session_id}.jsonl"
            media_type = "application/json"
        elif file_type == "schema":
            # Try both schema naming patterns
            filename = f"schema_output_{session_id}.json"
            if not os.path.exists(os.path.join(app.state.output_folder, filename)):
                filename = f"analysis_schema_{session_id}.json"
            media_type = "application/json"
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")

        file_path = os.path.join(app.state.output_folder, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type=media_type, filename=filename)
        else:
            raise HTTPException(status_code=404, detail="File not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/CONFIGURATION.md")
async def serve_configuration_guide():
    """Serve the configuration guide"""
    file_path = "CONFIGURATION.md"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Configuration guide not found")
    return FileResponse(file_path, media_type="text/markdown")


# RL Feedback Collection Endpoints
@app.post("/feedback/examples")
async def submit_example_feedback(request: Request):
    """Collect user feedback on generated examples - RL Training Data"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        example_index = data.get("example_index") 
        feedback_type = data.get("feedback_type")  # 'positive' or 'negative'
        example_data = data.get("example_data")
        timestamp = data.get("timestamp")
        
        print(f"🧠 [RL FEEDBACK] Session: {session_id}, Example {example_index}: {feedback_type}")
        
        # Use the feedback analyzer to store and analyze feedback
        success = feedback_analyzer.store_feedback(
            session_id=session_id,
            example_index=example_index,
            feedback_type=feedback_type,
            example_data=example_data
        )
        
        if success:
            print(f"✅ [RL] Feedback stored successfully for future prompt optimization")
            return {
                "success": True,
                "message": "Feedback recorded for RL training",
                "learning_active": True
            }
        else:
            raise Exception("Failed to store feedback in analyzer")
        
    except Exception as e:
        print(f"❌ [RL FEEDBACK ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.post("/feedback/prompt_description")
async def submit_prompt_feedback(request: Request):
    """Collect user feedback on generated prompt descriptions - RL Training Data"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        feedback_type = data.get("feedback_type")
        feedback_source = data.get("feedback_source", "prompt_description")
        document_class = data.get("document_class")
        prompt_text = data.get("prompt_text")
        detailed_feedback = data.get("detailed_feedback")
        
        print(f"🧠 [RL PROMPT FEEDBACK] Session: {session_id}, Document: {document_class}, Type: {feedback_type}")
        
        # Extract domain name from template
        domain_name = None
        try:
            template = db.get_template(document_class)
            domain_name = template.get("domain_name") if template else None
            print(f"🧠 [RL] Found domain for prompt feedback {document_class}: {domain_name}")
        except Exception as e:
            print(f"⚠️ [RL] Could not extract domain for prompt feedback {document_class}: {e}")
        
        # Create example data for prompt feedback storage
        example_data = {
            "document_class": document_class,
            "domain_name": domain_name,
            "feedback_source": feedback_source,
            "prompt_text": prompt_text,
            "timestamp": data.get("timestamp")
        }
        
        # Use the existing feedback analyzer to store prompt feedback
        success = feedback_analyzer.store_feedback(
            session_id=session_id,
            example_index="prompt", # Use "prompt" as a special index for prompt feedback
            feedback_type=feedback_type,
            example_data=example_data,
            detailed_feedback=detailed_feedback
        )
        
        if success:
            print(f"✅ [RL] Prompt feedback stored successfully for DSPy optimization")
            return {
                "success": True,
                "message": "Prompt description feedback recorded for RL training",
                "learning_active": True,
                "dspy_will_optimize": True
            }
        else:
            raise Exception("Failed to store prompt feedback in analyzer")
        
    except Exception as e:
        print(f"❌ [RL PROMPT FEEDBACK ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record prompt feedback: {str(e)}")


@app.post("/feedback/examples/detailed")
async def submit_detailed_feedback(request: Request):
    """Collect detailed user feedback with ratings and specific issues"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        example_index = data.get("example_index")
        detailed_feedback = data.get("detailed_feedback")
        example_data = data.get("example_data")
        
        print(f"🧠 [RL DETAILED FEEDBACK] Session: {session_id}, Example {example_index}")
        print(f"   Rating: {detailed_feedback.get('rating')}/5")
        print(f"   Issues: {detailed_feedback.get('issues', [])}")
        print(f"   Comments: {detailed_feedback.get('comments', 'N/A')[:50]}...")
        
        # Store detailed feedback using the analyzer
        success = feedback_analyzer.store_feedback(
            session_id=session_id,
            example_index=example_index,
            feedback_type="detailed",  # Mark as detailed feedback
            example_data=example_data,
            detailed_feedback=detailed_feedback
        )
        
        if success:
            print(f"✅ [RL] Detailed feedback stored for prompt learning")
            return {
                "success": True,
                "message": "Detailed feedback recorded for RL training",
                "learning_active": True,
                "next_optimization": "Feedback will improve future example generation"
            }
        else:
            raise Exception("Failed to store detailed feedback in analyzer")
        
    except Exception as e:
        print(f"❌ [RL DETAILED FEEDBACK ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record detailed feedback: {str(e)}")


@app.get("/feedback/stats/{document_class}")
async def get_feedback_stats(document_class: str):
    """Get RL learning statistics for a document class"""
    try:
        # Use the feedback analyzer to get comprehensive stats
        stats = feedback_analyzer.get_feedback_stats(document_class)
        
        # Get feedback analysis for optimization insights
        analysis = feedback_analyzer.analyze_feedback_for_document_class(document_class)
        
        return {
            "success": True,
            "document_class": document_class,
            "stats": stats,
            "analysis": analysis,
            "learning_insights": {
                "optimizations_available": len(analysis.get("optimization_suggestions", [])),
                "prompt_improvements": len(analysis.get("prompt_improvements", [])),
                "common_issues": analysis.get("common_issues", {}),
                "satisfaction_trend": "improving" if stats.get("satisfaction_rate", 0) > 0.7 else "needs_attention"
            },
            "message": f"RL analytics for {document_class} - Learning system active"
        }
        
    except Exception as e:
        print(f"❌ [RL STATS ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")


@app.get("/rl/debug/{document_class}")
async def rl_debug_feedback(document_class: str):
    """Debug endpoint to check RL feedback analysis"""
    try:
        # Reload feedback
        feedback_analyzer.load_stored_feedback()
        
        # Get analysis
        analysis = feedback_analyzer.analyze_feedback_for_document_class(document_class)
        
        return {
            "success": True,
            "document_class": document_class,
            "total_feedback_entries": len(feedback_analyzer.feedback_store),
            "feedback_entries": list(feedback_analyzer.feedback_store.keys()),
            "analysis": analysis,
            "debug_info": {
                "feedback_store_sample": {
                    k: {
                        "document_class": v.get('example_data', {}).get('document_class'),
                        "feedback_type": v.get('feedback_type'),
                        "has_detailed_feedback": bool(v.get('detailed_feedback')),
                        "comments": v.get('detailed_feedback', {}).get('comments', '')[:100] + "..." if v.get('detailed_feedback', {}).get('comments') else None
                    } 
                    for k, v in list(feedback_analyzer.feedback_store.items())[:3]
                }
            },
            "message": f"Debug info for {document_class}"
        }
        
    except Exception as e:
        print(f"❌ [RL DEBUG ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to debug feedback: {str(e)}")


@app.get("/rl/dashboard")
async def rl_dashboard(request: Request):
    """RL Learning Dashboard - shows feedback analytics"""
    try:
        if not hasattr(app.state, 'rl_feedback_store'):
            app.state.rl_feedback_store = []
            
        dashboard_data = {
            "total_sessions": len(set(f.get("session_id") for f in app.state.rl_feedback_store if f.get("session_id"))),
            "total_feedback": len(app.state.rl_feedback_store),
            "feedback_breakdown": {
                "simple": len([f for f in app.state.rl_feedback_store if f.get("type") == "simple_feedback"]),
                "detailed": len([f for f in app.state.rl_feedback_store if f.get("type") == "detailed_feedback"])
            },
            "recent_feedback": app.state.rl_feedback_store[-10:] if app.state.rl_feedback_store else []
        }
        
        return {
            "success": True,
            "dashboard_data": dashboard_data,
            "rl_enabled": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load RL dashboard: {str(e)}")


# Document class management endpoints
@app.get("/api/document_classes")
async def get_document_classes():
    """Get all document classes with statistics"""
    try:
        document_classes = get_all_document_classes()
        
        # Get statistics for each document class
        class_stats = []
        for doc_class in document_classes:
            # Get template info
            has_template = db.template_exists(doc_class)
            
            # Get feedback stats
            feedback_stats = feedback_analyzer.get_feedback_stats(doc_class)
            
            class_stats.append({
                "document_class": doc_class,
                "has_template": has_template,
                "feedback_count": feedback_stats.get('total', 0),
                "positive_feedback": feedback_stats.get('positive', 0),
                "negative_feedback": feedback_stats.get('negative', 0),
                "satisfaction_rate": feedback_stats.get('satisfaction_rate', 0),
                "average_rating": feedback_stats.get('average_rating')
            })
        
        return {
            "success": True,
            "document_classes": class_stats,
            "total_classes": len(document_classes)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document classes: {str(e)}")


@app.get("/rl/document_classes")
async def rl_document_classes_dashboard(request: Request):
    """RL Document Classes Dashboard"""
    try:
        return templates.TemplateResponse("rl_document_classes.html", {"request": request})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load document classes dashboard: {str(e)}")


# Domain management endpoints
@app.post("/register_domain")
async def register_domain(request: Request):
    """Register a new domain"""
    try:
        data = await request.json()
        domain_name = data.get("domain_name", "").strip()
        description = data.get("description", "").strip()
        
        if not domain_name:
            raise HTTPException(status_code=400, detail="Domain name is required")
        
        if db.domain_exists(domain_name):
            raise HTTPException(
                status_code=409,
                detail=f'A domain named "{domain_name}" already exists'
            )
        
        domain_id = db.register_domain(domain_name=domain_name, description=description)
        
        return {
            "success": True,
            "domain_id": domain_id,
            "message": f'Domain "{domain_name}" registered successfully!'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to register domain: {str(e)}"
        )


@app.get("/list_domains")
async def list_domains():
    """List all registered domains"""
    try:
        domains = db.list_domains()
        # Update domain statistics before returning
        db.update_domain_statistics()
        domains = db.list_domains()  # Get updated data
        
        return {"success": True, "domains": domains}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to list domains: {str(e)}"
        )


@app.get("/get_domain/{domain_name}")
async def get_domain(domain_name: str):
    """Get a specific domain by name"""
    try:
        domain = db.get_domain(domain_name)
        
        if not domain:
            raise HTTPException(
                status_code=404,
                detail=f'No domain found with name "{domain_name}"'
            )
        
        # Get templates in this domain
        templates = db.get_templates_by_domain(domain_name)
        domain["templates"] = templates
        
        return {"success": True, "domain": domain}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve domain: {str(e)}"
        )


@app.put("/update_domain/{domain_name}")
async def update_domain(domain_name: str, request: Request):
    """Update domain description"""
    try:
        data = await request.json()
        description = data.get("description", "").strip()
        
        if not db.domain_exists(domain_name):
            raise HTTPException(
                status_code=404,
                detail=f'No domain found with name "{domain_name}"'
            )
        
        success = db.update_domain(domain_name, description)
        
        if success:
            return {
                "success": True,
                "message": f'Domain "{domain_name}" updated successfully!'
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to update domain"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update domain: {str(e)}"
        )


@app.delete("/delete_domain/{domain_name}")
async def delete_domain(domain_name: str):
    """Delete a domain (only if no templates are assigned)"""
    try:
        success = db.delete_domain(domain_name)
        
        if success:
            # Clean up DSPy domain optimizations
            try:
                dspy_pipeline = get_dspy_pipeline()
                if dspy_pipeline:
                    cleanup_success = dspy_pipeline.delete_domain_optimizations(domain_name)
                    if cleanup_success:
                        print(f"🗑️ [DSPy] Cleaned up all optimizations for domain {domain_name}")
                    else:
                        print(f"🔍 [DSPy] No optimizations found for domain {domain_name}")
            except Exception as e:
                print(f"⚠️ [DSPy] Warning: Could not clean up domain optimizations: {e}")
            
            return {
                "success": True,
                "message": f'Domain "{domain_name}" and all associated optimizations deleted successfully!'
            }
        else:
            # Check if domain exists
            if not db.domain_exists(domain_name):
                raise HTTPException(
                    status_code=404,
                    detail=f'No domain found with name "{domain_name}"'
                )
            else:
                # Domain exists but has templates assigned
                templates = db.get_templates_by_domain(domain_name)
                raise HTTPException(
                    status_code=409,
                    detail=f'Cannot delete domain "{domain_name}" - it has {len(templates)} templates assigned to it'
                )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete domain: {str(e)}"
        )


@app.get("/api/domains")
async def get_domains_api():
    """API endpoint to get all domains (for dropdowns, etc.)"""
    try:
        domains = db.list_domains()
        
        # Return simplified format for UI components
        domain_list = [
            {
                "name": domain["domain_name"],
                "description": domain["description"],
                "template_count": domain.get("total_templates", 0)
            }
            for domain in domains
        ]
        
        return {
            "success": True,
            "domains": domain_list,
            "total_domains": len(domain_list)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get domains: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    # Get DSPy pipeline status
    dspy_status = {
        "enabled": DSPY_ENABLED,
        "compiled": False,
        "compilation_count": 0,
        "configured": False
    }
    
    if DSPY_ENABLED:
        try:
            pipeline = get_dspy_pipeline()
            dspy_status["compiled"] = pipeline.is_compiled
            dspy_status["compilation_count"] = pipeline.compilation_count
            
            # Check if DSPy is actually configured
            import dspy
            dspy_status["configured"] = dspy.settings.lm is not None
            
            # Add configuration hints
            if not dspy_status["configured"]:
                if config.provider == "azure_openai":
                    dspy_status["config_hint"] = "Set AZURE_OPENAI_API_KEY environment variable to enable DSPy"
                elif config.provider == "openai":
                    dspy_status["config_hint"] = "Set OPENAI_API_KEY environment variable to enable DSPy"
                else:
                    dspy_status["config_hint"] = f"DSPy not supported for provider: {config.provider}"
                    
        except Exception as e:
            dspy_status["error"] = str(e)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "provider": config.provider,
        "active_tasks": len(active_tasks),
        "rl_feedback_count": len(getattr(app.state, 'rl_feedback_store', [])),
        "rl_enabled": True,
        "dspy_status": dspy_status
    }


@app.get("/api/feedback/{document_class}")
async def get_feedback_for_document_class(document_class: str):
    """Get all feedback entries for a specific document class"""
    try:
        from docflash.rl_feedback_analyzer import feedback_analyzer
        
        # Get feedback from storage
        feedback_entries = []
        for feedback_id, feedback_data in feedback_analyzer.feedback_store.items():
            if feedback_data.get('example_data', {}).get('document_class') == document_class:
                # Add feedback_id to the response for deletion purposes
                feedback_entry = feedback_data.copy()
                feedback_entry['feedback_id'] = feedback_id
                feedback_entries.append(feedback_entry)
        
        # Sort by timestamp (newest first)
        feedback_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            "success": True,
            "document_class": document_class,
            "feedback": feedback_entries,
            "total_count": len(feedback_entries)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feedback: {str(e)}"
        )


@app.post("/api/feedback/master")
async def store_master_feedback(request: Request):
    """Store master feedback for a document class"""
    try:
        data = await request.json()
        document_class = data.get("document_class", "").strip()
        feedback_type = data.get("feedback_type", "")
        timestamp = data.get("timestamp", "")
        session_id = data.get("session_id", "")
        detailed_feedback = data.get("detailed_feedback")
        examples = data.get("examples")
        
        if not document_class or not feedback_type:
            raise HTTPException(
                status_code=400,
                detail="document_class and feedback_type are required"
            )
        
        from docflash.rl_feedback_analyzer import feedback_analyzer
        
        # Extract domain name from template
        domain_name = None
        try:
            template = db.get_template(document_class)
            domain_name = template.get("domain_name") if template else None
            print(f"🧠 [RL] Found domain for {document_class}: {domain_name}")
        except Exception as e:
            print(f"⚠️ [RL] Could not extract domain for {document_class}: {e}")
        
        # Create unique feedback ID for master feedback (use timestamp for uniqueness, not session)
        from datetime import datetime
        timestamp_id = int(datetime.now().timestamp() * 1000)  # milliseconds for uniqueness
        feedback_session_id = f"master_{document_class}_{timestamp_id}"
        
        # Prepare example data for storage with domain and complete examples
        example_data = {
            "document_class": document_class,
            "domain_name": domain_name,
            "feedback_source": "master",
            "session_id": session_id,
            "examples_count": len(examples) if examples else 0,
            "examples": examples if examples else [],
            "timestamp_id": timestamp_id
        }
        
        # Store the master feedback with document class based session ID
        success = feedback_analyzer.store_feedback(
            session_id=feedback_session_id,
            example_index=0,  # Use 0 for master feedback
            feedback_type=feedback_type,
            example_data=example_data,
            detailed_feedback=detailed_feedback
        )
        
        feedback_id = f"{feedback_session_id}_0"  # This will be the actual file name
        
        if success:
            print(f"🧠 [RL] Stored master feedback: {feedback_type} for document class: {document_class}")
            return {
                "success": True,
                "message": "Master feedback stored successfully",
                "feedback_id": feedback_id,
                "document_class": document_class
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to store master feedback"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store master feedback: {str(e)}"
        )


@app.delete("/api/feedback/{feedback_id}")
async def delete_feedback_entry(feedback_id: str):
    """Delete a specific feedback entry"""
    try:
        from docflash.rl_feedback_analyzer import feedback_analyzer
        
        # Check if feedback exists
        if feedback_id not in feedback_analyzer.feedback_store:
            raise HTTPException(
                status_code=404,
                detail=f"Feedback entry '{feedback_id}' not found"
            )
        
        # Get document class before deletion for logging
        document_class = feedback_analyzer.feedback_store[feedback_id].get('example_data', {}).get('document_class', 'unknown')
        
        # Delete from memory
        del feedback_analyzer.feedback_store[feedback_id]
        
        # Delete the file if it exists
        import os
        feedback_file = os.path.join(feedback_analyzer.storage_path, f"{feedback_id}.json")
        if os.path.exists(feedback_file):
            os.remove(feedback_file)
            print(f"🗑️ [RL] Deleted feedback file: {feedback_file}")
        
        print(f"🗑️ [RL] Deleted specific feedback entry: {feedback_id} for document class: {document_class}")
        
        return {
            "success": True,
            "message": f"Feedback entry deleted successfully",
            "deleted_feedback_id": feedback_id,
            "document_class": document_class
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete feedback entry: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True, log_level="info")