# 🏥 Medical Report Generator
![Python](https://img.shields.io/badge/Python-3.10-blue)
![LLaMA](https://img.shields.io/badge/LLaMA-3.3%2070B-green)
![FHIR](https://img.shields.io/badge/FHIR-R4%20Compliant-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

## 🎯 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mou11/medical-report-generator)

## 🔍 Overview
An end-to-end clinical report generation pipeline that generates structured medical reports from patient data, detects hallucinations using NLI, outputs FHIR R4 compliant JSON, and exports professional PDF reports.

## 📊 Results

| Report Type | BERTScore F1 | Safety Score |
|-------------|-------------|--------------|
| Radiology | 0.8869 | 0.75 |
| Discharge Summary | 0.9045 | 1.0 |
| Lab Report | 0.8129 | 0.375 |

## ✨ Features
- ✅ Generates 3 types of clinical reports: Radiology, Discharge Summary, Lab Report
- ✅ Hallucination detection using NLI (cross-encoder/nli-deberta-v3-base)
- ✅ FHIR R4 compliant JSON output (HL7 healthcare standard)
- ✅ BERTScore and ROUGE evaluation metrics
- ✅ Professional PDF export with quality assessment table
- ✅ FastAPI REST API for downstream integration

## 🏗️ Architecture
Patient Data → Report Generation (LLaMA 3.3 70B) → Hallucination Detection (DeBERTa NLI) → Evaluation (BERTScore + ROUGE) → FHIR R4 JSON → PDF Export

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq — LLaMA 3.3 70B |
| Hallucination Detection | cross-encoder/nli-deberta-v3-base |
| Evaluation | BERTScore + ROUGE |
| FHIR Output | HL7 R4 JSON |
| PDF Export | ReportLab |
| API | FastAPI |
| UI | Gradio |

## 🚀 How to Run
1. Get a free Groq API key at [console.groq.com](https://console.groq.com)
2. Open `app.py` in Google Colab
3. Replace `your-groq-api-key-here` with your actual key
4. Run all cells in order

## ⚠️ Medical Disclaimer
This system is for educational and research purposes only. It does not provide medical advice. Always consult a qualified healthcare professional for medical decisions.

## 📌 Project Status
✅ Report generation pipeline complete
✅ Hallucination detection implemented
✅ FHIR R4 compliant output
✅ PDF export working
✅ Gradio demo deployed on Hugging Face Spaces
