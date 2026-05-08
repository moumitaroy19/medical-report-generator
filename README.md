# Medical Report Generator
**Medical AI Portfolio — Project 4**

An end-to-end clinical report generation pipeline that generates structured medical reports from patient data, detects hallucinations using NLI, outputs FHIR R4 compliant JSON, and exports professional PDF reports.

## Live Demo
[HuggingFace Spaces](https://huggingface.co/spaces/mou11/medical-report-generator)

## Features
- Generates 3 types of clinical reports: Radiology, Discharge Summary, Lab Report
- Hallucination detection using NLI (cross-encoder/nli-deberta-v3-base)
- FHIR R4 compliant JSON output (HL7 healthcare standard)
- BERTScore and ROUGE evaluation metrics
- Professional PDF export with quality assessment table

## Tech Stack
| Component | Tool |
|---|---|
| LLM | Groq — LLaMA 3.3 70B |
| Hallucination Detection | cross-encoder/nli-deberta-v3-base |
| Evaluation | BERTScore + ROUGE |
| FHIR Output | HL7 R4 JSON |
| PDF Export | ReportLab |
| UI | Gradio |

## Results
| Report Type | BERTScore F1 | Safety Score |
|---|---|---|
| Radiology | 0.8869 | 0.75 |
| Discharge | 0.9045 | 1.0 |
| Lab | 0.8129 | 0.375 |

## Architecture
Patient Data → Report Generation (LLaMA 3.3 70B) → Hallucination Detection (DeBERTa NLI) → Evaluation (BERTScore + ROUGE) → FHIR R4 JSON → PDF Export
