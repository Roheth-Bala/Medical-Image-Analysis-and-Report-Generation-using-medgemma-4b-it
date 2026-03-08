# Chest X-ray Analysis

Modern local chest X-ray reporting app built with:
- `Gradio` UI
- `Ollama` multimodal inference (MedGemma)
- formatted in-app report preview
- formatted PDF download

The app runs locally and generates one structured report per uploaded X-ray.

## Features

- Single-pass model inference
- Clean report sections:
  - Findings
  - Impression
  - Recommendations
  - Focused probability estimates
- Priority tagging:
  - `ROUTINE`
  - `HIGH`
  - `STAT`
- Button-gated outputs:
  - `View Report`
  - `Download Report`
- PDF export with readable clinical formatting

## Project Structure

- `app.py` - Gradio app + inference + formatting + PDF generation
- `requirements.txt` - Python dependencies
- `.env.example` - safe environment template for GitHub
- `sample_images/` - test images folder

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama server:

```bash
ollama serve
```

4. Ensure model exists locally:

```bash
ollama ls
```

Expected model example:
- `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M`

5. Run the app:

```bash
python app.py
```

Open:
- `http://127.0.0.1:7860`

## Using the App

1. Upload a chest X-ray.
2. (Optional) Add clinical context.
3. Click `Generate Report`.
4. After processing completes:
   - Click `View Report` to render the report on screen.
   - Click `Download Report` to get the PDF.

## Test Inputs (Two Lung Images)

Place your two images in:
- `sample_images/normal lung.png`
- `sample_images/pneumonia lung.png`

These names are referenced so testing is consistent across environments.

## GitHub-safe Environment Handling

- `.env` is ignored by git.
- `.env.example` is committed.
- Keep secrets/token values out of tracked files.

Current `.gitignore` already includes:
- `.env`
- `venv/`
- `__pycache__/`

## Clinical Safety Note

This tool is decision support only.  
It does **not** replace clinical judgment, radiologist review, or institutional protocols.

"# Medical-Image-Analysis-and-Report-Generation-using-medgemma-4b-it" 
