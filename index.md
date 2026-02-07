---
layout: single
title: ""
permalink: /
author_profile: true   # hide the big name card on the left
---

---

## Technical Skills {#skills}

- Programming Languages: Python, JavaScript
- ML / AI: PyTorch, TensorFlow, Scikit-learn, Computer Vision, Image Processing
- LLMs & RAG: OpenAI, Hugging Face, LangChain, Prompt Engineering, RAG, AI Agents
- Generative AI: GANs, Diffusion Models
- Data & Experimentation: NumPy, Pandas, Feature Engineering, Optuna, TensorBoard
- Vector Search & Databases: FAISS, Pinecone, ChromaDB, PostgreSQL, SQLite, MongoDB
- Cloud & MLOps: AWS, Docker, FastAPI, CI/CD (GitHub Actions, GitLab)
- Automation & Workflows: n8n, AI Workflow Orchestration
- Software Engineering: REST APIs, Node.js
- Developer Tools: Git, GitHub, GitLab, Linux, Cursor, VS Code


**Tools & Libraries**
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
---

**Tech Support & IT Operations**

- Troubleshooting hardware, software, and network issues
- Linux & Windows system administration
- User support, incident management, and ticketing systems
- Network fundamentals (TCP/IP, DNS, DHCP, VPN)
- Server monitoring, and system diagnostics
- Software installation, configuration, and updates
- Remote support tools and end-user assistance
- Documentation and technical knowledge base creation
  
---


## Projects {#projects}

### 1. AI-Powered Job Matching & Application Assistant *(01/26 – 02/26)* 
-	Designed an end-to-end automated pipeline in n8n that monitors incoming job emails, extracts job descriptions, and processes them for AI-based evaluation.
-	Built a Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and Pinecone vector database to compare job requirements against a candidate’s CV using semantic similarity search.
-	Designed a rule-based + AI agent to classify opportunities into Apply / Maybe / Skip using skill alignment, seniority, and requirement matching.


> **Tech:**  n8n, OpenAI (Embeddings + LLMs), Pinecone, Retrieval-Augmented Generation (RAG), Semantic Similarity Search, AI Agents, Python, Email APIs (IMAP/Gmail), Multilingual Processing (DE ↔ EN translation), Rule-based decision logic.


### 2. Enterprise Multi-Domain RAG AI Agent with Real-Time Streaming *(01/2026)*
-	Designed and implemented a *real-time AI agent* powered by OpenAI LLM that answers enterprise policy and FAQ questions using Retrieval-Augmented Generation (RAG), producing grounded responses with source citations from internal knowledge bases.
-	Developed an automated routing layer that enables the AI agent to classify user queries and dynamically select the appropriate domain-specific knowledge base (HR, Finance, IT), demonstrating tool-selection and agentic decision-making.
-	Deployed the production-ready AI Agent on Render (backend) and Vercel (frontend).


> **Tech:** Python, FastAPI, OpenAI LLMs, LangChain, RAG, ChromaDB, SQLite, Vercel, Render, GitHub CI/CD  
> **Role:** Sole developer responsible for end-to-end design and implementation.

<a href="https://github.com/devo002/Enterprise-RAG-AI-Agent" 
   target="_blank" 
   class="project-link">
  <img src="/assets/icons/github.svg" alt="GitHub" class="github-icon">
  View Code
</a>



### 3. Log Analysis System using LLMs on AWS *(12/2025 - 01/2026)*
-	Developed an LLM-powered log analysis service using FastAPI and AWS Bedrock (Claude) to automatically detect error patterns, infer root causes, and generate recommendations.
-	Reduced manual debugging effort by enabling automatic log clustering and failure diagnosis using LLM reasoning.
-	Designed and implemented an event-driven log analysis pipeline using AWS S3 + AWS Lambda, with automated result persistence and CloudWatch-based observability.


> **Tech:** Python, AWS Bedrock (Claude), S3, Lambda, CloudWatch, FastAPI, JSON, LLM  
> **Role:** Sole developer responsible for end-to-end design and implementation.

<a href="https://github.com/devo002/LLM-log-analyzer" 
   target="_blank" 
   class="project-link">
  <img src="/assets/icons/github.svg" alt="GitHub" class="github-icon">
  View Code
</a>

---

### 4. Retrieval-Augmented Generation (RAG) Chatbot with Nvidia (10/2025 - 11/2025)
-	Designed and implemented an intelligent RAG chatbot leveraging NVIDIA Llama models integrated with FastAPI and Gradio for interactive document-based conversations. 
-	Engineered a long-term conversational context storage system using a database layer, enabling the chatbot to maintain continuity in discussions over uploaded PDF and HTML documents across sessions. 
-	Containerized the system with Docker to support reliable deployment and scalability.

> **Tech:** Python, NVIDIA Llama, RAG, SQLite, FAISS, Docker, FastAPI

<a href="https://github.com/devo002/RAG-chatbot-nvidia" 
   target="_blank" 
   class="project-link">
  <img src="/assets/icons/github.svg" alt="GitHub" class="github-icon">
  View Code
</a>

---

### 5. Handwriting Generation (AFFGANwriting Improvement) (06/2025 - 11/2025):
-	Redesigned the handwriting style encoder using a custom transformer model, improving writing realism and increasing user study pick-rates by 40%. 
-	Developed an interactive web application using Streamlit, enabling users to input text to generate and experiment with handwriting styles dynamically.


> **Tech:** PyTorch, GANs, Hugging face, CNNs, Transformer Models, Streamlit
> **Role:** End-to-end design, training, evaluation, to show that modern backbone models improve the realism of a handwriting style.

<a href="https://github.com/devo002/Handwriting_generation" 
   target="_blank" 
   class="project-link">
  <img src="/assets/icons/github.svg" alt="GitHub" class="github-icon">
  View Code
</a>

---

### 6. Semi-Supervised Learning for Image Classification in Visual Inspection of Wound Coils (12/2023 - 12/2024):
-	Implemented the FixMatch semi-supervised learning (SSL) algorithm with the Dinov2L backbone to accurately classify defects in coil winding datasets, achieving a macro-average F1 score of 90% on the test set.
-	Conducted hyperparameter optimization using Optuna, identifying optimal configurations to maximize model performance.
-	Delivered comprehensive, reproducible documentation and code, publicly available on GitHub. 

> **Tech:** PyTorch, SSL, Hugging face, DINOv2-L, EfficientNetV2-L, FixMatch, MixMatch, Optuna, Tensorboard, Wandb  
> **Role:** Implemented a semi-supervised FixMatch-based multi-label classification pipeline, improving defect detection performance to 90% macro F1, outperforming the fully supervised baseline (87% macro F1).

<a href="https://github.com/devo002/Semi-supervised-Learning-for-Visual-Coil-winding-defect-detection" 
   target="_blank" 
   class="project-link">
  <img src="/assets/icons/github.svg" alt="GitHub" class="github-icon">
  View Code
</a>

---

## Education
### Master of Science (M.Sc.) in Artificial Intelligence
**Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**, Germany  
*2022 – 2025*

### Bachelor of Technology in Computer Science
**Federal University of Technology Minna**, Nigeria  
*2014 – 2018*

---

## Work Experience
### FAU- Department of Chemistry and Pharmarcy - Erlangen, Germany
IT HIWI | 06/2023 – 09/2025

-	Secured and streamlined network infrastructure (LAN, WAN, VPN, firewalls, routers, switches), improving connectivity, reducing latency by 30%, and enhancing business continuity.
-	Administered Active Directory and Group Policy Objects (GPO), managing  user accounts, enforcing role-based access, and improving compliance with internal security policies.
-	Provided Tier 1–3 technical support to end-users, resolving hardware, software, and network issues within SLA timelines, achieving a 95% user satisfaction rating.
-	Conducted IT audits, compliance checks, and vulnerability assessments, ensuring 100% adherence to internal policies, and other regulatory requirements.
-	Managed IT procurement, asset tracking, and lifecycle management.
-	Coordinated with third-party vendors and service providers for escalated technical issues, maintaining 99% SLA compliance and ensuring uninterrupted business operations. 
-	Implemented network segmentation, VPN access controls, and MFA authentication, significantly improving security posture for remote and hybrid work environments.


---

If you’d like to know more about any of these projects or collaborate, feel free to reach out at  
📧 **davidmayowaonaiyekan@gmail.com** or connect on [LinkedIn](https://www.linkedin.com/in/david-mayowa-onaiyekan-01b436122).
