import os
import json
import time
import re
import uuid
from datetime import datetime

from groq import Groq
from transformers import pipeline
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER
import gradio as gr


# ── Groq Client ──────────────────────────────────────────────
class GroqClientManager:
    def __init__(self):
        self.keys = []
        self.current_index = 0
        self._load_keys()

    def _load_keys(self):
        for i in range(1, 6):
            key = os.environ.get(f"GROQ_KEY_{i}")
            if key:
                self.keys.append(key)
        if not self.keys:
            raise ValueError("No Groq API keys found. Add GROQ_KEY_1 to Space secrets.")
        print(f"Loaded {len(self.keys)} Groq key(s)")

    def get_client(self):
        return Groq(api_key=self.keys[self.current_index])

    def rotate(self):
        self.current_index = (self.current_index + 1) % len(self.keys)

    def chat(self, messages, max_tokens=1500, temperature=0.3):
        for _ in range(len(self.keys)):
            try:
                client = self.get_client()
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    self.rotate()
                    time.sleep(2)
                else:
                    raise e
        raise RuntimeError("All Groq API keys exhausted.")


groq_manager = GroqClientManager()

print("Loading NLI model...")
nli_pipeline = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-base"
)
print("NLI model loaded.")


# ── Report Prompts ────────────────────────────────────────────
REPORT_PROMPTS = {
    "radiology": """You are a radiologist writing a formal radiology report.
Given the following patient data, generate a structured radiology report.

Patient Data:
{patient_data}

Generate a report with these exact sections:
CLINICAL INDICATION:
TECHNIQUE:
FINDINGS:
IMPRESSION:

Be specific, clinical, and only use information provided. Do not invent findings.""",

    "discharge": """You are a hospital physician writing a discharge summary.
Given the following patient data, generate a structured discharge summary.

Patient Data:
{patient_data}

Generate a report with these exact sections:
PATIENT INFORMATION:
ADMISSION DIAGNOSIS:
HOSPITAL COURSE:
DISCHARGE DIAGNOSIS:
DISCHARGE MEDICATIONS:
FOLLOW-UP INSTRUCTIONS:

Only use information provided. Do not invent medications or diagnoses.""",

    "lab": """You are a clinical pathologist writing a laboratory report.
Given the following patient data, generate a structured lab report.

Patient Data:
{patient_data}

Generate a report with these exact sections:
TEST ORDERED:
SPECIMEN:
RESULTS:
REFERENCE RANGES:
INTERPRETATION:
RECOMMENDATION:

Only use information provided. Do not invent lab values."""
}


# ── Core Functions ────────────────────────────────────────────
def generate_report(patient_data, report_type):
    patient_data_str = "\n".join([f"{k}: {v}" for k, v in patient_data.items()])
    prompt = REPORT_PROMPTS[report_type].format(patient_data=patient_data_str)
    response = groq_manager.chat([{"role": "user", "content": prompt}])
    return {
        "report_type": report_type,
        "report_text": response.strip(),
        "patient_data": patient_data,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def extract_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences
            if len(s.strip()) > 20 and not s.strip().isupper() and ":" not in s[:25]]


def check_hallucination(report):
    source_text = " ".join([f"{k} is {v}." for k, v in report["patient_data"].items()])
    sentences = extract_sentences(report["report_text"])
    if not sentences:
        return {"error": "No checkable sentences found."}

    results = []
    hallucination_count = 0

    for sentence in sentences:
        nli_input = f"{source_text} [SEP] {sentence}"
        prediction = nli_pipeline(nli_input, truncation=True, max_length=512)
        label = prediction[0]["label"].lower()
        score = prediction[0]["score"]

        if "entail" in label:
            status = "supported"
        elif "contradict" in label:
            status = "hallucinated"
            hallucination_count += 1
        else:
            status = "unverified"
            if score > 0.80:
                hallucination_count += 0.5

        results.append({"sentence": sentence, "status": status, "confidence": round(score, 4)})

    total = len(sentences)
    hallucination_rate = round(hallucination_count / total, 4) if total > 0 else 0

    return {
        "report_type": report["report_type"],
        "total_claims": total,
        "hallucination_rate": hallucination_rate,
        "safety_score": round(1 - hallucination_rate, 4),
        "claim_results": results
    }


def evaluate_report(report):
    reference = " ".join([f"{k} is {v}." for k, v in report["patient_data"].items()])
    hypothesis = report["report_text"]

    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)

    return {
        "bertscore": {
            "precision": round(P[0].item(), 4),
            "recall":    round(R[0].item(), 4),
            "f1":        round(F1[0].item(), 4)
        },
        "rouge": {
            "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
            "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
            "rougeL": round(rouge_scores["rougeL"].fmeasure, 4)
        }
    }


def create_fhir_report(report, hallucination_result):
    report_type_codes = {
        "radiology": {"code": "18748-4", "display": "Diagnostic imaging study"},
        "discharge": {"code": "18842-5", "display": "Discharge summary"},
        "lab":       {"code": "11502-2", "display": "Laboratory report"}
    }
    code_info = report_type_codes.get(report["report_type"], {"code": "unknown", "display": "Clinical Report"})

    return {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                                   "code": code_info["code"],
                                   "display": code_info["display"]}]}],
        "code": {"text": f"{report['report_type'].capitalize()} Report"},
        "subject": {"reference": f"Patient/{str(uuid.uuid4())}",
                    "display": report["patient_data"].get("name", "Unknown")},
        "effectiveDateTime": report["generated_at"],
        "issued": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conclusion": report["report_text"],
        "extension": [
            {"url": "https://medical-ai-portfolio.dev/fhir/hallucination-score",
             "valueDecimal": hallucination_result.get("hallucination_rate", 0)},
            {"url": "https://medical-ai-portfolio.dev/fhir/safety-score",
             "valueDecimal": hallucination_result.get("safety_score", 1)},
            {"url": "https://medical-ai-portfolio.dev/fhir/total-claims-checked",
             "valueInteger": hallucination_result.get("total_claims", 0)}
        ]
    }


def export_pdf(report, hallucination_result, eval_result):
    output_path = f"/tmp/{report['report_type']}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    title_style   = ParagraphStyle("title",   parent=styles["Title"],   fontSize=16, spaceAfter=6,  alignment=TA_CENTER)
    subtitle_style= ParagraphStyle("subtitle",parent=styles["Normal"],  fontSize=9,  spaceAfter=12, alignment=TA_CENTER, textColor=colors.grey)
    section_style = ParagraphStyle("section", parent=styles["Heading2"],fontSize=11, spaceBefore=12,spaceAfter=4, textColor=colors.HexColor("#1a1a2e"))
    body_style    = ParagraphStyle("body",    parent=styles["Normal"],  fontSize=9,  spaceAfter=6,  leading=14)
    label_style   = ParagraphStyle("label",   parent=styles["Normal"],  fontSize=9,  textColor=colors.grey)

    story = []
    story.append(Paragraph("Medical Report Generator", title_style))
    story.append(Paragraph(f"Medical AI Portfolio — Project 4 | Generated: {report['generated_at']}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"{report['report_type'].upper()} REPORT", section_style))
    story.append(Spacer(1, 6))

    patient_table_data = [[Paragraph(f"<b>{k.replace('_',' ').title()}</b>", label_style),
                            Paragraph(str(v), body_style)]
                           for k, v in report["patient_data"].items()]
    patient_table = Table(patient_table_data, colWidths=[1.8*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,-1), colors.HexColor("#f0f0f0")),
        ("GRID",       (0,0),(-1,-1),0.5, colors.lightgrey),
        ("VALIGN",     (0,0),(-1,-1),"TOP"),
        ("PADDING",    (0,0),(-1,-1),6),
    ]))
    story.append(Paragraph("Patient Information", section_style))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Generated Report", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))
    for line in report["report_text"].split("\n"):
        if line.strip():
            if line.strip().isupper() or line.strip().endswith(":"):
                story.append(Paragraph(f"<b>{line.strip()}</b>", body_style))
            else:
                story.append(Paragraph(line.strip(), body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Quality Assessment", section_style))
    quality_data = [
        ["Metric", "Value"],
        ["Total Claims Checked",  str(hallucination_result.get("total_claims", 0))],
        ["Hallucination Rate",    str(hallucination_result.get("hallucination_rate", 0))],
        ["Safety Score",          str(hallucination_result.get("safety_score", 0))],
        ["BERTScore F1",          str(eval_result["bertscore"]["f1"])],
        ["ROUGE-1",               str(eval_result["rouge"]["rouge1"])],
        ["ROUGE-2",               str(eval_result["rouge"]["rouge2"])],
        ["ROUGE-L",               str(eval_result["rouge"]["rougeL"])],
    ]
    quality_table = Table(quality_data, colWidths=[2.5*inch, 2*inch])
    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
        ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0),(-1,-1),0.5, colors.lightgrey),
        ("BACKGROUND", (0,1),(-1,-1),colors.HexColor("#f9f9f9")),
        ("PADDING",    (0,0),(-1,-1),6),
    ]))
    story.append(quality_table)
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph("Generated by Medical Report Generator | Moumita Roy | Medical AI Portfolio Project 4", subtitle_style))
    doc.build(story)
    return output_path


# ── Pipeline ──────────────────────────────────────────────────
def run_pipeline(name, age, sex, chief_complaint, vitals, history, imaging, labs, report_type):
    patient_data = {
        "name": name, "age": age, "sex": sex,
        "chief_complaint": chief_complaint, "vitals": vitals,
        "history": history, "imaging": imaging, "labs": labs
    }
    patient_data = {k: v for k, v in patient_data.items() if v and str(v).strip()}

    report       = generate_report(patient_data, report_type)
    h_result     = check_hallucination(report)
    e_result     = evaluate_report(report)
    fhir         = create_fhir_report(report, h_result)
    pdf_path     = export_pdf(report, h_result, e_result)

    claim_breakdown = "\n".join([
        f"[{c['status'].upper():12}] ({c['confidence']}) {c['sentence'][:90]}..."
        for c in h_result.get("claim_results", [])
    ])

    hallucination_summary = f"""Total Claims Checked : {h_result['total_claims']}
Hallucination Rate   : {h_result['hallucination_rate']}
Safety Score         : {h_result['safety_score']}

Claim-level Breakdown:
{claim_breakdown}"""

    metrics_summary = f"""BERTScore  —  P: {e_result['bertscore']['precision']}  R: {e_result['bertscore']['recall']}  F1: {e_result['bertscore']['f1']}
ROUGE-1    :  {e_result['rouge']['rouge1']}
ROUGE-2    :  {e_result['rouge']['rouge2']}
ROUGE-L    :  {e_result['rouge']['rougeL']}"""

    return (
        report["report_text"],
        hallucination_summary,
        metrics_summary,
        json.dumps(fhir, indent=2),
        pdf_path
    )


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Medical Report Generator") as demo:
    gr.Markdown("# Medical Report Generator")
    gr.Markdown("Medical AI Portfolio — Project 4 | Moumita Roy | [GitHub](https://github.com/moumitaroy19/medical-report-generator)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Information")
            name            = gr.Textbox(label="Full Name",        value="John Doe")
            age             = gr.Textbox(label="Age",              value="58")
            sex             = gr.Textbox(label="Sex",              value="Male")
            chief_complaint = gr.Textbox(label="Chief Complaint",  value="Chest pain and shortness of breath for 2 days")
            vitals          = gr.Textbox(label="Vitals",           value="BP 145/90, HR 88, RR 18, Temp 37.1C, SpO2 96%")
            history         = gr.Textbox(label="Medical History",  value="Hypertension, Type 2 Diabetes, smoker for 20 years")
            imaging         = gr.Textbox(label="Imaging",          value="Chest X-ray ordered, mild cardiomegaly noted")
            labs            = gr.Textbox(label="Lab Results",      value="WBC 11.2, HGB 13.4, Troponin 0.02, BNP 210")
            report_type     = gr.Dropdown(choices=["radiology", "discharge", "lab"],
                                          value="radiology", label="Report Type")
            submit_btn      = gr.Button("Generate Report", variant="primary")

        with gr.Column():
            gr.Markdown("### Output")
            report_output        = gr.Textbox(label="Generated Report",       lines=12)
            hallucination_output = gr.Textbox(label="Hallucination Analysis", lines=10)
            metrics_output       = gr.Textbox(label="Evaluation Metrics",     lines=6)
            fhir_output          = gr.Textbox(label="FHIR R4 JSON",           lines=10)
            pdf_output           = gr.File(label="Download PDF Report")

    submit_btn.click(
        fn=run_pipeline,
        inputs=[name, age, sex, chief_complaint, vitals, history, imaging, labs, report_type],
        outputs=[report_output, hallucination_output, metrics_output, fhir_output, pdf_output]
    )

demo.launch()
