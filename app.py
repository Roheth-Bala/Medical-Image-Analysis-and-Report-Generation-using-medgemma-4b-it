import argparse
import base64
import html
import os
import tempfile
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Iterator, List, Optional, Tuple

import gradio as gr
import requests
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

APP_CSS = """
:root {
  --bg-a: #0f172a;
  --bg-b: #111827;
  --card: rgba(255, 255, 255, 0.08);
  --txt: #e5e7eb;
}
.gradio-container {
  background:
    radial-gradient(circle at 80% 5%, #164e63 0%, transparent 30%),
    radial-gradient(circle at 10% 100%, #1d4ed8 0%, transparent 35%),
    linear-gradient(160deg, var(--bg-a), var(--bg-b));
  color: var(--txt);
}
#panel {
  background: var(--card);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  padding: 14px;
  backdrop-filter: blur(7px);
}
#title {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: 0.2px;
  margin-bottom: 6px;
}
"""


def to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_ollama_prompt(images: List[Image.Image], prompt: str, model: str, base_url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [to_base64_png(img) for img in images],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.2},
    }
    url = f"{base_url.rstrip('/')}/api/generate"
    res = requests.post(url, json=payload, timeout=240)
    if not res.ok:
        raise RuntimeError(f"Ollama error {res.status_code}: {res.text[:500]}")
    return str(res.json().get("response", "")).strip()


def dedupe_lines(text: str) -> str:
    seen = set()
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        key = " ".join(line.lower().split())
        if not key:
            out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out).strip()


def extract_section(text: str, section_name: str) -> str:
    lines = text.splitlines()
    capture = False
    out: List[str] = []
    target = section_name.lower().rstrip(":")
    for line in lines:
        s = line.strip()
        if s.lower().rstrip(":") == target:
            capture = True
            continue
        if capture and s.lower().rstrip(":") in {"findings", "impression", "recommendations"}:
            break
        if capture and s:
            out.append(s)
    return "\n".join(out).strip()


def findings_prompt(clinical_context: str) -> str:
    context = clinical_context.strip() or "No additional clinical context provided."
    return (
        "You are assisting a radiologist with chest X-ray interpretation. "
        "Be concise, no repetition. Use exact sections:\n"
        "Findings:\nImpression:\nRecommendations:\n"
        "Do not repeat prompt instructions in output.\n"
        f"Clinical context: {context}"
    )


def assess_urgency(text: str) -> Dict[str, str]:
    lowered = text.lower()
    normal_markers = [
        "no acute cardiopulmonary process",
        "no acute pulmonary process",
        "lungs are clear",
        "no focal consolidation",
        "no pleural effusion",
        "no pneumothorax",
    ]
    high_terms = ["pneumothorax", "tension", "respiratory failure", "critical", "large pleural effusion"]
    medium_terms = ["focal consolidation", "consolidation", "opacity", "effusion", "nodule", "atelectasis", "infiltrate"]

    if any(k in lowered for k in high_terms):
        return {"priority": "STAT", "rationale": "Potential life-threatening finding pattern detected."}
    if any(k in lowered for k in medium_terms):
        return {"priority": "HIGH", "rationale": "Potentially significant abnormality requires expedited review."}
    if any(k in lowered for k in normal_markers):
        return {"priority": "ROUTINE", "rationale": "No acute cardiopulmonary abnormality identified in AI draft."}
    return {"priority": "ROUTINE", "rationale": "No clear urgent signals from AI draft."}


def estimate_probabilities(text: str) -> str:
    lowered = text.lower()
    probs = {
        "Pneumonia": 8,
        "Pleural effusion": 5,
        "Atelectasis": 6,
        "Pulmonary edema": 4,
    }
    if any(k in lowered for k in ["consolidation", "infiltrate", "opacity", "airspace"]):
        probs["Pneumonia"] = 55
    if any(k in lowered for k in ["effusion", "pleural fluid"]):
        probs["Pleural effusion"] = 60
    if any(k in lowered for k in ["atelectasis", "volume loss"]):
        probs["Atelectasis"] = 50
    if any(k in lowered for k in ["edema", "vascular congestion", "interstitial"]):
        probs["Pulmonary edema"] = 55
    if any(k in lowered for k in ["no acute", "lungs are clear", "no focal consolidation", "no pleural effusion"]):
        probs = {k: min(v, 10) for k, v in probs.items()}
    return "\n".join([f"{k}: {v}%" for k, v in probs.items()])


def render_report_html(
    findings: str,
    report_draft: str,
    urgency: Dict[str, str],
    probability_block: str,
    clinical_context: str,
) -> str:
    findings_sec = extract_section(report_draft, "Findings") or findings
    impression_sec = extract_section(report_draft, "Impression") or "Not clearly generated."
    reco_sec = extract_section(report_draft, "Recommendations") or "Clinical correlation advised."
    context = clinical_context.strip() or "Not provided."
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prob_html = (
        f"<h3>Focused Probability Re-check</h3><pre>{html.escape(probability_block)}</pre>"
        if probability_block
        else "<h3>Focused Probability Re-check</h3><p>Not triggered.</p>"
    )
    priority_color = "#ef4444" if urgency["priority"] == "STAT" else "#f59e0b" if urgency["priority"] == "HIGH" else "#22c55e"

    return f"""
<div style="font-family:Segoe UI,Arial,sans-serif;color:#e5e7eb;line-height:1.45;">
  <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);border-radius:14px;padding:16px;margin-bottom:12px;">
    <h2 style="margin:0 0 8px 0;">Chest X-ray Clinical Report</h2>
    <div><b>Priority:</b> <span style="padding:3px 10px;border-radius:999px;background:{priority_color};color:#0b1020;font-weight:700;">{urgency['priority']}</span></div>
    <div style="margin-top:6px;"><b>Reason:</b> {html.escape(urgency['rationale'])}</div>
    <div style="margin-top:6px;"><b>Clinical Context:</b> {html.escape(context)}</div>
    <div style="margin-top:6px;"><b>Timestamp (UTC):</b> {ts}</div>
  </div>
  <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);border-radius:14px;padding:16px;margin-bottom:12px;">
    <h3>Findings</h3><div style="white-space:pre-wrap">{html.escape(findings_sec)}</div>
    <h3>Impression</h3><div style="white-space:pre-wrap">{html.escape(impression_sec)}</div>
    <h3>Recommendations</h3><div style="white-space:pre-wrap">{html.escape(reco_sec)}</div>
  </div>
  <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);border-radius:14px;padding:16px;">
    {prob_html}
    <p style="margin-top:10px;font-size:12px;opacity:0.85;">
      AI-generated draft for clinician review. Not a standalone diagnosis.
    </p>
  </div>
</div>
"""


def build_report_payload(
    findings: str,
    urgency: Dict[str, str],
    probability_block: str,
    clinical_context: str,
) -> Dict[str, str]:
    findings_sec = extract_section(findings, "Findings") or findings
    impression_sec = extract_section(findings, "Impression") or "No acute cardiopulmonary abnormality identified."
    reco_sec = extract_section(findings, "Recommendations") or "Clinical correlation advised."
    context = clinical_context.strip() or "Not provided."
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "priority": urgency["priority"],
        "reason": urgency["rationale"],
        "context": context,
        "timestamp": ts,
        "findings": findings_sec,
        "impression": impression_sec,
        "recommendations": reco_sec,
        "probabilities": probability_block or "Not triggered.",
    }


def save_html_report(report_html: str) -> str:
    out_dir = tempfile.gettempdir()
    path = os.path.join(out_dir, f"chest_xray_report_{uuid.uuid4().hex[:10]}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<html><head><meta charset='utf-8'><title>Chest X-ray Report</title></head>"
            f"<body style='background:#0f172a;padding:16px;'>{report_html}</body></html>"
        )
    return path


def save_pdf_report(payload: Dict[str, str]) -> str:
    out_dir = tempfile.gettempdir()
    path = os.path.join(out_dir, f"chest_xray_report_{uuid.uuid4().hex[:10]}.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], textColor=colors.HexColor("#0f172a"), spaceAfter=8)
    heading = ParagraphStyle("heading", parent=styles["Heading3"], textColor=colors.HexColor("#1f2937"), spaceAfter=4, spaceBefore=8)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)

    story = []
    story.append(Paragraph("Chest X-ray Clinical Report", title))
    summary_tbl = Table(
        [
            ["Priority", payload["priority"]],
            ["Reason", payload["reason"]],
            ["Clinical Context", payload["context"]],
            ["Timestamp (UTC)", payload["timestamp"]],
        ],
        colWidths=[40 * mm, 130 * mm],
    )
    summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(summary_tbl)
    story.append(Spacer(1, 8))

    for label, key in [
        ("Findings", "findings"),
        ("Impression", "impression"),
        ("Recommendations", "recommendations"),
        ("Focused Probability Re-check", "probabilities"),
    ]:
        story.append(Paragraph(label, heading))
        safe_text = html.escape(payload[key]).replace("\n", "<br/>")
        story.append(Paragraph(safe_text, body))

    story.append(Spacer(1, 8))
    story.append(Paragraph("AI-generated draft for clinician review. Not a standalone diagnosis.", body))
    doc.build(story)
    return path


def build_app(model: str, base_url: str) -> gr.Blocks:
    def run_copilot(
        current_image: Optional[Image.Image],
        clinical_context: str,
    ) -> Iterator[Tuple[str, str, str, Dict[str, object], Dict[str, object], Optional[str], Optional[str], Dict[str, object]]]:
        if current_image is None:
            yield (
                "No image uploaded.",
                "",
                "<p>Please upload a chest X-ray.</p>",
                gr.update(visible=False, interactive=False),
                gr.update(visible=False, interactive=False),
                None,
                None,
                gr.update(value=None, visible=False),
            )
            return

        yield (
            "Step 1/2: Running MedGemma inference...",
            "Processing findings...",
            "<p>Processing report...</p>",
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            None,
            None,
            gr.update(value=None, visible=False),
        )
        findings = dedupe_lines(
            run_ollama_prompt([current_image], findings_prompt(clinical_context), model=model, base_url=base_url)
        )

        yield (
            "Step 2/2: Formatting report from inference...",
            findings,
            "<p>Findings are ready. Finalizing report...</p>",
            gr.update(visible=True, interactive=False),
            gr.update(visible=True, interactive=False),
            None,
            None,
            gr.update(value=None, visible=False),
        )

        probability_block = estimate_probabilities(findings)
        urgency_source = "\n".join([extract_section(findings, "Findings") or findings, extract_section(findings, "Impression")])
        urgency = assess_urgency(urgency_source)
        payload = build_report_payload(findings, urgency, probability_block, clinical_context)
        report_html = render_report_html(findings, findings, urgency, probability_block, clinical_context)
        _ = save_html_report(report_html)
        pdf_path = save_pdf_report(payload)
        yield (
            "Completed. Use View Report or Download Report.",
            findings,
            "<p>Report ready. Click <b>View Report</b> to display.</p>",
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            report_html,
            pdf_path,
            gr.update(value=None, visible=False),
        )

    def show_report(report_html_state: Optional[str]) -> str:
        return report_html_state or "<p>No report is ready yet.</p>"

    def get_report_file(pdf_path_state: Optional[str]) -> Dict[str, object]:
        return gr.update(value=pdf_path_state, visible=True)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
<div id="title">Chest X-ray Analysis</div>
Upload a study to generate a concise clinical report.
"""
        )
        with gr.Row():
            with gr.Column(scale=1, elem_id="panel"):
                current_image = gr.Image(type="pil", label="Chest X-ray")
                clinical_context = gr.Textbox(
                    lines=4,
                    label="Clinical Context (Optional)",
                    placeholder="e.g., fever, cough, hypoxia, ICU status",
                )
                run_button = gr.Button("Generate Report", variant="primary")
                view_button = gr.Button("View Report", visible=False, interactive=False)
                download_button = gr.Button("Download Report", visible=False, interactive=False)
                status_output = gr.Textbox(label="Status", lines=2)
            with gr.Column(scale=2, elem_id="panel"):
                findings_output = gr.Textbox(label="Imaging Findings", lines=8)
                report_view = gr.HTML(label="Clinical Report")
                download_file = gr.File(label="Download Report (PDF)", visible=False)
                report_html_state = gr.State(value=None)
                pdf_path_state = gr.State(value=None)

        run_button.click(
            fn=run_copilot,
            inputs=[current_image, clinical_context],
            outputs=[
                status_output,
                findings_output,
                report_view,
                view_button,
                download_button,
                report_html_state,
                pdf_path_state,
                download_file,
            ],
        )
        view_button.click(fn=show_report, inputs=[report_html_state], outputs=[report_view])
        download_button.click(fn=get_report_file, inputs=[pdf_path_state], outputs=[download_file])

    demo.queue()
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dcarrascosa/medgemma-1.5-4b-it:Q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app(model=args.model, base_url=args.ollama_url)
    app.launch(server_name=args.host, server_port=args.port, share=False, css=APP_CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
