layout: single
title: ""
permalink: /
author_profile: true
<style>
.demo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin: 16px 0 24px;
}
.demo-card {
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  padding: 14px 16px;
  background: #fafafa;
}
.demo-card h4 { margin: 0 0 4px; font-size: 15px; }
.demo-card p { margin: 0 0 10px; font-size: 13px; color: #555; line-height: 1.4; }
.demo-tag { font-size: 11px; background: #e8f0fe; color: #1a56db; border-radius: 4px; padding: 2px 7px; display: inline-block; margin-bottom: 8px; }
.demo-btn {
  display: inline-block;
  font-size: 12px;
  padding: 5px 12px;
  border: 1px solid #1a56db;
  border-radius: 5px;
  color: #1a56db;
  text-decoration: none;
  margin-right: 6px;
}
.demo-btn:hover { background: #e8f0fe; }
.proj-compact { border-left: 3px solid #e0e0e0; padding: 6px 0 6px 14px; margin: 12px 0; }
.proj-compact h3 { margin: 0 0 4px; font-size: 15px; }
.proj-compact p { margin: 0 0 6px; font-size: 13px; color: #444; }
.proj-compact .tech { font-size: 11px; color: #777; }
</style>

🔴 Live Demos {#demos}
<div class="demo-grid">
  <div class="demo-card">
    <span class="demo-tag">Agentic AI</span>
    <h4>Smart Dispatcher</h4>
    <p>LangGraph agent that triages support tickets with a 6-node self-correction loop — no re-prompting needed when a fix fails</p>
    <a class="demo-btn" href="https://smart-dispatcher-agent.onrender.com/" target="_blank">▶ Open demo</a>
    <a class="demo-btn" href="https://empire-smart-dispatcher-production.up.railway.app/" target="_blank">Code</a>
  </div>
  <div class="demo-card">
    <span class="demo-tag">RAG</span>
    <h4>RAG Doc Chat</h4>
    <p>Upload PDFs and chat with them — streaming answers with inline source citations and an automated RAGAS eval pipeline</p>
    <a class="demo-btn" href="https://exemplary-manifestation-production-dcca.up.railway.app/" target="_blank">▶ Open demo</a>
    <a class="demo-btn" href="https://github.com/devo002/RAG-doc-chat" target="_blank">Code</a>
  </div>
  <div class="demo-card">
    <span class="demo-tag">Full-stack</span>
    <h4>Budget Tracker</h4>
    <p>Personal finance tracker with weekly spending limits, real-time summaries, and mobile support — deployed on Vercel + Neon</p>
    <a class="demo-btn" href="https://budget-tracker-lac-six.vercel.app/dashboard" target="_blank">▶ Open demo</a>
    <a class="demo-btn" href="https://github.com/devo002/Budget-Tracker" target="_blank">Code</a>
  </div>
</div>

Technical Skills {#skills}

Languages: Python, JavaScript, TypeScript
ML / AI: PyTorch, TensorFlow, Scikit-learn, Optuna, TensorBoard
Computer Vision: OpenCV, Image Processing
Generative AI: OpenAI, Anthropic, Llama, Hugging Face, LangChain, LangGraph, LlamaIndex, RAG, Prompt Engineering, MCP
Databases & Vector Stores: PostgreSQL, MySQL, MongoDB, Qdrant, Pinecone, ChromaDB
Cloud & MLOps: AWS, Docker, FastAPI, CI/CD (GitHub Actions, GitLab)
Automation: n8n, AI Workflow Orchestration


Education
M.Sc. Artificial Intelligence — Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Germany (2022 – 2025)
B.Tech Computer Science — Federal University of Technology Minna, Nigeria (2014 – 2018)

Key Projects {#projects}
AI Ticket Bridge Agent — Teams → Jira (04/2026)
Multi-step agent that classifies 70+ Microsoft Teams messages in under 4 Claude API calls, cross-references HubSpot ARR, deduplicates against Jira, and streams decisions to a full-stack review UI.
Python Claude Flask Jira API SSE Docker
  Code

AI-Powered Job Matching Assistant (01/26 – 02/26)
n8n pipeline that monitors job emails, embeds descriptions, and scores them against a CV using Pinecone semantic search — classifying each as Apply / Maybe / Skip.
n8n OpenAI Pinecone RAG Python IMAP

LLM Log Analyzer on AWS (12/2025 – 01/2026)
Event-driven pipeline on AWS Bedrock + Lambda that auto-clusters logs, infers root causes, and persists results — replacing manual debugging.
Python AWS Bedrock Lambda S3 CloudWatch FastAPI
  Code

RAG Chatbot with NVIDIA Llama (10/2025 – 11/2025)
Document-grounded chatbot with persistent conversation history across PDF and HTML uploads, containerised with Docker.
Python NVIDIA Llama FAISS FastAPI Docker
  Code

Research Experience {#research-experience}
Pattern Recognition Lab, FAU (06/2025 – 11/2025)
Replaced the VGG19 backbone in a handwriting GAN with a Transformer encoder — boosting perceptual pick-rates by 40% and OCR accuracy by 20% via a teacher-student alignment framework.
PyTorch GANs Transformers Streamlit
  Code

FAPS, FAU — Semi-supervised Defect Detection (12/2023 – 12/2024)
Built a FixMatch + DINOv2-L SSL pipeline on unlabelled coil-winding images, reaching 90% macro F1 on multi-label defect classification (vs. 87% supervised baseline).
PyTorch DINOv2-L FixMatch Optuna Weights & Biases
  Code

Computer Vision Projects, FAU (04/2025 – 09/2025)
Built selective search object detection end-to-end and a video face recognition system combining MTCNN, FaceNet embeddings, and from-scratch k-NN / k-means.
Python OpenCV ResNet-50 Scikit-learn

Tools & Libraries
<div class="skill-grid">
  <img src="/assets/icons/pytorch.svg" alt="Pytorch" />
  <img src="/assets/icons/tensorflow.svg" alt="Tensorflow" />
  <img src="/assets/icons/opencv.svg" alt="Open CV" />
  <img src="/assets/icons/jupyter.svg" alt="Jupyter" />
  <img src="/assets/icons/python.svg" alt="Python" />
  <img src="/assets/icons/javascript.svg" alt="Javascript" />
  <img src="/assets/icons/nodedotjs.svg" alt="NodeJS" />
  <img src="/assets/icons/mongodb.svg" alt="MongoDB" />
  <img src="/assets/icons/huggingface.svg" alt="Hugging Face" />
  <img src="/assets/icons/git.svg" alt="Git" />
  <img src="/assets/icons/github.svg" alt="Github" />
  <img src="/assets/icons/numpy.svg" alt="Numpy" />
  <img src="/assets/icons/docker.svg" alt="Docker" />
  <img src="/assets/icons/langchain.svg" alt="Langchain" />
  <img src="/assets/icons/optuna.svg" alt="Optuna" />
</div>

📧 davidmayowaonaiyekan@gmail.com · LinkedIn