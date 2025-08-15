#!/usr/bin/env python3
"""
PDF OCR Pipeline for Doc Flash
Converts PDF -> Images -> OCR -> Markdown with configurable LLM providers
Supports Azure OpenAI, OpenAI, Google Gemini, Ollama, and vLLM service
"""
import asyncio
import base64
import concurrent.futures
import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import aiohttp
import cv2
import numpy as np
import pdf2image
import pikepdf
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageStat

# Import configuration
try:
    from .config import config
    from .providers import ModelFactory

    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False
    print("⚠️ Provider configuration not available, falling back to default service")


class PDFOCRPipeline:
    def __init__(
        self, service_url=None, max_concurrent=10, use_configured_provider=True
    ):
        # Get service URL from config or use default
        if service_url is None:
            service_url = os.getenv(
                "VLLM_SERVICE_URL", "http://localhost:8000"
            )

        self.service_url = service_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.use_configured_provider = use_configured_provider and PROVIDERS_AVAILABLE

        # Initialize OCR model if using configured providers
        if self.use_configured_provider and PROVIDERS_AVAILABLE:
            try:
                # Check if user explicitly wants vLLM
                ocr_provider = os.getenv("OCR_PROVIDER", config.provider).lower()
                if ocr_provider == "vllm":
                    print("📡 Using vLLM service for OCR (explicitly configured)")
                    self.use_configured_provider = False
                else:
                    self.ocr_model = ModelFactory.create_model(config.ocr_config)
                    self.provider_type = config.ocr_config.provider
                    print(f"🤖 Using {self.provider_type} for OCR processing")
            except Exception as e:
                print(f"⚠️ Failed to initialize configured OCR provider: {e}")
                print("📡 Falling back to vLLM service")
                self.use_configured_provider = False
        else:
            print("📡 Using vLLM service for OCR")

    def is_blank(self, img):
        """Check if image page is blank"""
        stat = ImageStat.Stat(img)
        return max(stat.mean) == 0

    def process_pdf_page(self, pdf_path, page_idx):
        """Process a single PDF page to detect orientation"""
        try:
            images = pdf2image.convert_from_path(
                pdf_path, first_page=page_idx + 1, last_page=page_idx + 1, dpi=300
            )

            if not images:
                return 0

            img = images[0]
            if self.is_blank(img):
                return 0

            img_cv = np.array(img)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            try:
                osd = pytesseract.image_to_osd(gray)
                angle = int(re.search("(?<=Rotate: )\\d+", osd).group(0))
            except Exception:
                angle = 0

            return angle
        except Exception as e:
            print(f"Error processing page {page_idx}: {e}")
            return 0

    def pdf_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF to individual page images with orientation correction"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Open PDF and detect orientation
            pdf = pikepdf.open(pdf_path)
            total_pages = len(pdf.pages)

            print(f"📄 Processing {total_pages} pages for orientation detection...")

            # Detect page orientations in parallel
            page_angles = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.process_pdf_page, pdf_path, i): i
                    for i in range(total_pages)
                }
                for future in concurrent.futures.as_completed(futures):
                    page_idx = futures[future]
                    try:
                        angle = future.result()
                        if angle and angle != 0:
                            page_angles[page_idx] = angle
                    except Exception as e:
                        print(f"Error processing page {page_idx}: {e}")

            # Apply rotations if needed
            if page_angles:
                print(f"🔄 Applying rotations to {len(page_angles)} pages...")
                for page_idx, angle in page_angles.items():
                    if angle != 0:
                        page = pdf.pages[page_idx]
                        page.rotate(angle % 360)

                # Save corrected PDF
                corrected_pdf_path = os.path.join(output_dir, "corrected_document.pdf")
                pdf.save(corrected_pdf_path)
                pdf_path = corrected_pdf_path

            pdf.close()

            # Convert to images
            print(f"🖼️ Converting PDF to images at 300 DPI...")
            images = pdf2image.convert_from_path(pdf_path, dpi=300)

            image_paths = []
            for i, img in enumerate(images):
                if not self.is_blank(img):
                    image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                    img.save(image_path, "PNG")
                    image_paths.append(image_path)
                    print(f"✅ Saved page {i+1} -> {image_path}")
                else:
                    print(f"⚪ Skipped blank page {i+1}")

            return image_paths

        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            return []

    async def convert_image_to_markdown_openai(
        self, image_path: str, page_num: int
    ) -> Tuple[int, str]:
        """Convert image to markdown using OpenAI/Azure OpenAI GPT-4V"""

        DOCEXT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Follow these formatting guidelines:

**Text Formatting:**
- Return tables in HTML format
- Return equations in LaTeX representation  
- Use ☐ and ☑ for check boxes
- Preserve paragraph structure and line breaks

**Special Elements:**
- Watermarks: <watermark>OFFICIAL COPY</watermark>
- Page numbers: <page_number>14</page_number> or <page_number>9/22</page_number>
- Images: If caption present, use <img>caption text</img>. If no caption, add brief description <img>description of image</img>

**Handwritten Content:**
- Clear handwriting: Transcribe normally
- Unclear handwriting: Transcribe readable parts, mark unclear sections as [unclear]
- Signatures: Mark as [SIGNATURE] or [SIGNATURE: name] if name is legible
- Handwritten dates: Transcribe if clear, otherwise use [DATE: partial_date] format

**Stamps and Seals:**
- Official stamps: [STAMP: description]
- Notary seals: [NOTARY SEAL]
- Embossed elements: [EMBOSSED: description]

**Quality Guidelines:**
- Focus on clearly readable text
- Do not attempt to transcribe illegible or heavily stylized elements
- Mark ambiguous content appropriately rather than producing garbled output
- Maintain document structure and hierarchy"""

        try:
            # Encode image to base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            # Create payload for OpenAI/Azure OpenAI
            payload = {
                "model": config.ocr_config.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a document OCR assistant. Focus on clear text extraction and proper markup of special elements.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DOCEXT_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": config.ocr_config.max_tokens,
                "temperature": config.ocr_config.temperature,
            }

            # Make API call based on provider type
            if self.provider_type == "azure_openai":
                # Azure OpenAI API call
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {await self._get_azure_token()}",
                }
                url = f"{config.ocr_config.azure_endpoint}openai/deployments/{config.ocr_config.azure_deployment_name or config.ocr_config.model_id}/chat/completions?api-version={config.ocr_config.azure_api_version}"
            else:
                # OpenAI API call
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.ocr_config.openai_api_key}",
                }
                url = f"{config.ocr_config.openai_base_url or 'https://api.openai.com/v1'}/chat/completions"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        markdown = result["choices"][0]["message"]["content"]
                        markdown = self.cleanup_ocr_artifacts(markdown)
                        print(
                            f"✅ OCR completed for page {page_num} using {self.provider_type}"
                        )
                        return page_num, markdown
                    else:
                        error_text = await response.text()
                        print(
                            f"❌ OCR failed for page {page_num}: {response.status} - {error_text}"
                        )
                        return page_num, f"[OCR FAILED FOR PAGE {page_num}]"

        except Exception as e:
            print(f"❌ OCR error for page {page_num}: {e}")
            return page_num, f"[OCR ERROR FOR PAGE {page_num}: {str(e)}]"

    async def _get_azure_token(self):
        """Get Azure AD token for authentication"""
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            return token_provider()
        except Exception as e:
            print(f"Failed to get Azure token: {e}")
            return None

    async def convert_image_to_markdown_gemini(
        self, image_path: str, page_num: int
    ) -> Tuple[int, str]:
        """Convert image to markdown using Google Gemini"""

        # Import Google AI here to avoid dependency if not needed
        try:
            import google.generativeai as genai
        except ImportError:
            return (
                page_num,
                f"[OCR ERROR FOR PAGE {page_num}: Google AI package not installed]",
            )

        DOCEXT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Follow these formatting guidelines:

**Text Formatting:**
- Return tables in HTML format
- Return equations in LaTeX representation  
- Use ☐ and ☑ for check boxes
- Preserve paragraph structure and line breaks

**Special Elements:**
- Watermarks: <watermark>OFFICIAL COPY</watermark>
- Page numbers: <page_number>14</page_number> or <page_number>9/22</page_number>
- Images: If caption present, use <img>caption text</img>. If no caption, add brief description <img>description of image</img>

**Handwritten Content:**
- Clear handwriting: Transcribe normally
- Unclear handwriting: Transcribe readable parts, mark unclear sections as [unclear]
- Signatures: Mark as [SIGNATURE] or [SIGNATURE: name] if name is legible
- Handwritten dates: Transcribe if clear, otherwise use [DATE: partial_date] format

**Stamps and Seals:**
- Official stamps: [STAMP: description]
- Notary seals: [NOTARY SEAL]
- Embossed elements: [EMBOSSED: description]

**Quality Guidelines:**
- Focus on clearly readable text
- Do not attempt to transcribe illegible or heavily stylized elements
- Mark ambiguous content appropriately rather than producing garbled output
- Maintain document structure and hierarchy"""

        try:
            # Configure Gemini
            genai.configure(api_key=config.ocr_config.google_api_key)

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Create Gemini model
            model = genai.GenerativeModel(config.ocr_config.model_id)

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=config.ocr_config.temperature,
                max_output_tokens=config.ocr_config.max_tokens,
            )

            # Create content with text and image
            prompt_parts = [
                DOCEXT_PROMPT,
                {"mime_type": "image/png", "data": image_data},
            ]

            # Generate response
            response = model.generate_content(
                prompt_parts, generation_config=generation_config
            )

            if response and response.text:
                markdown = response.text
                markdown = self.cleanup_ocr_artifacts(markdown)
                print(f"✅ OCR completed for page {page_num} using Gemini")
                return page_num, markdown
            else:
                print(f"❌ No response from Gemini for page {page_num}")
                return page_num, f"[OCR FAILED FOR PAGE {page_num}]"

        except Exception as e:
            print(f"❌ Gemini OCR error for page {page_num}: {e}")
            return page_num, f"[OCR ERROR FOR PAGE {page_num}: {str(e)}]"

    async def convert_image_to_markdown_ollama(
        self, image_path: str, page_num: int
    ) -> Tuple[int, str]:
        """Convert image to markdown using Ollama"""

        DOCEXT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Follow these formatting guidelines:

**Text Formatting:**
- Return tables in HTML format
- Return equations in LaTeX representation  
- Use ☐ and ☑ for check boxes
- Preserve paragraph structure and line breaks

**Special Elements:**
- Watermarks: <watermark>OFFICIAL COPY</watermark>
- Page numbers: <page_number>14</page_number> or <page_number>9/22</page_number>
- Images: If caption present, use <img>caption text</img>. If no caption, add brief description <img>description of image</img>

**Handwritten Content:**
- Clear handwriting: Transcribe normally
- Unclear handwriting: Transcribe readable parts, mark unclear sections as [unclear]
- Signatures: Mark as [SIGNATURE] or [SIGNATURE: name] if name is legible
- Handwritten dates: Transcribe if clear, otherwise use [DATE: partial_date] format

**Stamps and Seals:**
- Official stamps: [STAMP: description]
- Notary seals: [NOTARY SEAL]
- Embossed elements: [EMBOSSED: description]

**Quality Guidelines:**
- Focus on clearly readable text
- Do not attempt to transcribe illegible or heavily stylized elements
- Mark ambiguous content appropriately rather than producing garbled output
- Maintain document structure and hierarchy"""

        try:
            # Import requests for Ollama API
            import requests
            import base64

            # Encode image to base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            # Create Ollama API payload
            payload = {
                "model": config.ocr_config.model_id,
                "prompt": f"[img]{base64_image}[/img]\n\n{DOCEXT_PROMPT}",
                "stream": False,
                "options": {
                    "temperature": config.ocr_config.temperature,
                }
            }

            # Make API call to Ollama
            response = requests.post(
                f"{config.ocr_config.ollama_base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minute timeout for vision models
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("response"):
                markdown = result["response"]
                markdown = self.cleanup_ocr_artifacts(markdown)
                print(f"✅ OCR completed for page {page_num} using Ollama")
                return page_num, markdown
            else:
                print(f"❌ No response from Ollama for page {page_num}")
                return page_num, f"[OCR FAILED FOR PAGE {page_num}]"

        except Exception as e:
            print(f"❌ Ollama OCR error for page {page_num}: {e}")
            return page_num, f"[OCR ERROR FOR PAGE {page_num}: {str(e)}]"

    async def convert_image_to_markdown(
        self, session: aiohttp.ClientSession, image_path: str, page_num: int
    ) -> Tuple[int, str]:
        """Convert single image to markdown using configured provider or fallback service"""

        # Use configured provider if available
        if self.use_configured_provider:
            if self.provider_type in ["azure_openai", "openai"]:
                return await self.convert_image_to_markdown_openai(image_path, page_num)
            elif self.provider_type == "gemini":
                return await self.convert_image_to_markdown_gemini(image_path, page_num)
            elif self.provider_type == "ollama":
                return await self.convert_image_to_markdown_ollama(image_path, page_num)

        # Fallback to original vLLM service
        return await self.convert_image_to_markdown_vllm(session, image_path, page_num)

    async def convert_image_to_markdown_vllm(
        self, session: aiohttp.ClientSession, image_path: str, page_num: int
    ) -> Tuple[int, str]:
        """Convert single image to markdown using original vLLM service (fallback)"""

        DOCEXT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Follow these formatting guidelines:

**Text Formatting:**
- Return tables in HTML format
- Return equations in LaTeX representation  
- Use ☐ and ☑ for check boxes
- Preserve paragraph structure and line breaks

**Special Elements:**
- Watermarks: <watermark>OFFICIAL COPY</watermark>
- Page numbers: <page_number>14</page_number> or <page_number>9/22</page_number>
- Images: If caption present, use <img>caption text</img>. If no caption, add brief description <img>description of image</img>

**Handwritten Content:**
- Clear handwriting: Transcribe normally
- Unclear handwriting: Transcribe readable parts, mark unclear sections as [unclear]
- Signatures: Mark as [SIGNATURE] or [SIGNATURE: name] if name is legible
- Handwritten dates: Transcribe if clear, otherwise use [DATE: partial_date] format

**Stamps and Seals:**
- Official stamps: [STAMP: description]
- Notary seals: [NOTARY SEAL]
- Embossed elements: [EMBOSSED: description]

**Quality Guidelines:**
- Focus on clearly readable text
- Do not attempt to transcribe illegible or heavily stylized elements
- Mark ambiguous content appropriately rather than producing garbled output
- Maintain document structure and hierarchy"""

        async with self.semaphore:
            try:
                # Encode image to base64
                with open(image_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")

                payload = {
                    "model": "nanonets-ocr",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a document OCR assistant. Focus on clear text extraction and proper markup of special elements.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                                {"type": "text", "text": DOCEXT_PROMPT},
                            ],
                        },
                    ],
                    "max_tokens": 16000,
                    "temperature": 0.0,
                    "stop": ["\\U\\U\\U\\U\\U", "\\C\\U\\U\\U"],
                }

                async with session.post(
                    f"{self.service_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        markdown = result["choices"][0]["message"]["content"]
                        markdown = self.cleanup_ocr_artifacts(markdown)

                        print(
                            f"✅ OCR completed for page {page_num} using vLLM service"
                        )
                        return page_num, markdown
                    else:
                        error_text = await response.text()
                        print(
                            f"❌ OCR failed for page {page_num}: {response.status} - {error_text}"
                        )
                        return page_num, f"[OCR FAILED FOR PAGE {page_num}]"

            except Exception as e:
                print(f"❌ Exception during OCR for page {page_num}: {e}")
                return page_num, f"[OCR ERROR FOR PAGE {page_num}: {str(e)}]"

    def cleanup_ocr_artifacts(self, text: str) -> str:
        """Clean up OCR artifacts and garbled output"""
        # Remove excessive repeated Unicode escape sequences
        text = re.sub(
            r"(\\[UC]){5,}.*", "\n[SIGNATURE/STAMP AREA]\n", text, flags=re.DOTALL
        )

        # Remove other patterns of repeated characters
        text = re.sub(r"([A-Z])\1{10,}", "[ILLEGIBLE TEXT]", text)
        text = re.sub(r"([\\\/\-_=]){8,}", "[ILLEGIBLE MARKINGS]", text)

        # Clean up excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n\s*\n", "\n\n\n", text)
        text = re.sub(r" {4,}", " ", text)

        return text.strip()

    def convert_to_natural_language(self, markdown: str) -> str:
        """Convert structured tags to natural language descriptions"""
        text = markdown

        # Handle signatures
        text = re.sub(r"\[SIGNATURE:\s*(.*?)\]", r"[SIGNED BY: \1]", text)
        text = re.sub(r"\[SIGNATURE\]", "[SIGNATURE PRESENT]", text)

        # Handle watermarks
        text = re.sub(
            r"<watermark>(.*?)</watermark>", r"[WATERMARK: \1]", text, flags=re.DOTALL
        )

        # Remove OCR-detected page numbers (we use our own **PAGE NUMBER X** headers)
        text = re.sub(
            r"<page_number>(.*?)</page_number>", "", text, flags=re.DOTALL
        )

        # Handle image descriptions
        text = re.sub(r"<img>(.*?)</img>", r"[IMAGE: \1]", text, flags=re.DOTALL)

        # Handle handwritten content markers
        text = re.sub(r"\[unclear\]", "[UNCLEAR HANDWRITING]", text)
        text = re.sub(r"\[DATE:\s*(.*?)\]", r"[HANDWRITTEN DATE: \1]", text)

        # Handle stamps and seals
        text = re.sub(r"\[STAMP:\s*(.*?)\]", r"[OFFICIAL STAMP: \1]", text)
        text = re.sub(r"\[NOTARY SEAL\]", "[NOTARY SEAL PRESENT]", text)
        text = re.sub(r"\[EMBOSSED:\s*(.*?)\]", r"[EMBOSSED ELEMENT: \1]", text)

        # Handle checkboxes
        text = text.replace("☐", "[EMPTY CHECKBOX]")
        text = text.replace("☑", "[CHECKED CHECKBOX]")
        text = text.replace("☒", "[MARKED CHECKBOX]")

        # Handle LaTeX equations
        text = re.sub(r"\$\$(.*?)\$\$", r"\n[EQUATION: \1]\n", text, flags=re.DOTALL)
        text = re.sub(r"\$(.*?)\$", r"[MATH EXPRESSION: \1]", text)

        # Convert HTML tables to markdown
        text = self.convert_html_tables_to_markdown(text)

        # Clean up multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def convert_html_tables_to_markdown(self, text: str) -> str:
        """Convert HTML tables to markdown tables"""
        table_pattern = r"<table.*?</table>"
        tables = re.findall(table_pattern, text, re.DOTALL | re.IGNORECASE)

        for html_table in tables:
            try:
                soup = BeautifulSoup(html_table, "html.parser")
                table = soup.find("table")

                if not table:
                    continue

                markdown_table = []
                rows = table.find_all("tr")

                for i, row in enumerate(rows):
                    cells = row.find_all(["td", "th"])
                    if cells:
                        row_data = []
                        for cell in cells:
                            cell_text = cell.get_text(strip=True)
                            cell_text = cell_text.replace("|", "\\|")
                            row_data.append(cell_text)

                        markdown_row = "| " + " | ".join(row_data) + " |"
                        markdown_table.append(markdown_row)

                        if i == 0:
                            separator = (
                                "| " + " | ".join(["---"] * len(row_data)) + " |"
                            )
                            markdown_table.append(separator)

                if markdown_table:
                    markdown_table_str = "\n".join(markdown_table)
                    text = text.replace(html_table, f"\n\n{markdown_table_str}\n\n")

            except Exception:
                text = text.replace(html_table, "[TABLE - CONVERSION FAILED]")

        return text

    async def _process_images_dynamic_batching(
        self, image_paths: List[str], progress_callback=None
    ) -> Dict[int, str]:
        """Process images with dynamic batching and retry logic"""

        # Create queues for task management
        pending_queue = asyncio.Queue()
        results = {}
        completed_count = 0
        total_images = len(image_paths)

        # Populate the queue with image tasks
        for i, image_path in enumerate(image_paths):
            page_num = i + 1  # 1-indexed page numbers
            await pending_queue.put((image_path, page_num))

        print(
            f"🚀 Starting dynamic OCR processing with {self.max_concurrent} concurrent workers"
        )

        async with aiohttp.ClientSession() as session:
            # Create worker tasks
            workers = []
            for worker_id in range(self.max_concurrent):
                worker = asyncio.create_task(
                    self._ocr_worker(session, pending_queue, results, worker_id)
                )
                workers.append(worker)

            # Monitor progress
            while completed_count < total_images:
                await asyncio.sleep(0.1)  # Check every 100ms
                current_completed = len(results)

                if current_completed > completed_count:
                    completed_count = current_completed
                    if progress_callback:
                        await progress_callback(
                            f"Completed {completed_count}/{total_images} pages"
                        )
                    print(
                        f"📊 Progress: {completed_count}/{total_images} pages completed"
                    )

            # Cancel all workers
            for worker in workers:
                worker.cancel()

            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

        return results

    async def _ocr_worker(
        self,
        session: aiohttp.ClientSession,
        queue: asyncio.Queue,
        results: Dict[int, str],
        worker_id: int,
    ):
        """Worker that processes OCR tasks with retry logic"""

        while True:
            try:
                # Get next task from queue (with timeout to avoid hanging)
                try:
                    image_path, page_num = await asyncio.wait_for(
                        queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No more tasks available
                    break

                # Process the image with retry logic
                success = False
                max_retries = 3
                base_delay = 1.0  # Start with 1 second delay

                for attempt in range(max_retries + 1):
                    try:
                        page_num_result, markdown = (
                            await self.convert_image_to_markdown(
                                session, image_path, page_num
                            )
                        )

                        # Check if result indicates failure
                        if "[OCR FAILED" in markdown or "[OCR ERROR" in markdown:
                            raise Exception(f"OCR processing failed: {markdown}")

                        # Success - store result
                        results[page_num] = markdown
                        success = True
                        print(f"✅ Worker {worker_id} completed page {page_num}")
                        break

                    except Exception as e:
                        error_msg = str(e).lower()

                        # Check if it's a rate limit error
                        is_rate_limit = any(
                            keyword in error_msg
                            for keyword in [
                                "rate limit",
                                "quota",
                                "too many requests",
                                "429",
                                "throttle",
                            ]
                        )

                        if attempt < max_retries:
                            # Calculate exponential backoff delay
                            if is_rate_limit:
                                delay = base_delay * (
                                    3**attempt
                                )  # More aggressive backoff for rate limits
                                print(
                                    f"⚠️ Worker {worker_id} hit rate limit on page {page_num}, "
                                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                                )
                            else:
                                delay = base_delay * (
                                    2**attempt
                                )  # Standard exponential backoff
                                print(
                                    f"⚠️ Worker {worker_id} error on page {page_num}: {e}, "
                                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                                )

                            await asyncio.sleep(delay)
                        else:
                            # Final attempt failed
                            print(
                                f"❌ Worker {worker_id} failed page {page_num} after {max_retries + 1} attempts: {e}"
                            )
                            results[page_num] = (
                                f"[OCR FAILED FOR PAGE {page_num} AFTER {max_retries + 1} ATTEMPTS: {str(e)}]"
                            )

                # Mark task as done
                queue.task_done()

            except asyncio.CancelledError:
                # Worker is being cancelled, exit gracefully
                break
            except Exception as e:
                print(f"❌ Worker {worker_id} unexpected error: {e}")
                break

    async def process_pdf_to_markdown(
        self, pdf_path: str, progress_callback=None
    ) -> str:
        """Complete pipeline: PDF -> Images -> OCR -> Markdown"""

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Step 1: Convert PDF to images
                if progress_callback:
                    await progress_callback("Converting PDF to images...")

                image_paths = self.pdf_to_images(pdf_path, temp_dir)

                if not image_paths:
                    return "❌ Failed to convert PDF to images"

                print(f"📄 Successfully extracted {len(image_paths)} pages")

                # Step 2: Process images with dynamic OCR batching
                if progress_callback:
                    await progress_callback(
                        f"Processing {len(image_paths)} pages with OCR..."
                    )

                page_results = await self._process_images_dynamic_batching(
                    image_paths, progress_callback
                )

                # Step 3: Combine results in page order
                if progress_callback:
                    await progress_callback("Combining pages into final document...")

                combined_markdown = []

                for page_num in sorted(page_results.keys()):
                    markdown = page_results[page_num]

                    # Add page header
                    page_header = f"\n\n**PAGE NUMBER {page_num}**\n\n"
                    combined_markdown.append(page_header)

                    # Add page content
                    combined_markdown.append(markdown)

                final_markdown = "".join(combined_markdown)

                # Step 4: Convert to natural language
                if progress_callback:
                    await progress_callback("Converting to natural language format...")

                natural_language = self.convert_to_natural_language(final_markdown)

                print(f"✅ OCR pipeline completed successfully!")
                print(f"📏 Final document: {len(natural_language)} characters")

                return natural_language

            except Exception as e:
                error_msg = f"❌ OCR Pipeline failed: {str(e)}"
                print(error_msg)
                return error_msg


# Async wrapper for Flask
def run_pdf_ocr_pipeline(pdf_path: str, progress_callback=None) -> str:
    """Synchronous wrapper for the async OCR pipeline"""

    async def _run():
        pipeline = PDFOCRPipeline(max_concurrent=10)
        return await pipeline.process_pdf_to_markdown(pdf_path, progress_callback)

    # Run the async function
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run())
        loop.close()
        return result
    except Exception as e:
        return f"❌ Pipeline execution failed: {str(e)}"