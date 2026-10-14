---
layout: single
title: ""
permalink: /
author_profile: true   # hide the big name card on the left # grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
---


<style>
  .demo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 18px;
  }

  .demo-tag {
    align-self: flex-start;
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
    background: #1a1a28;
    border: 1px solid #2a2a3a;
    color: #a8bfff;
    border-radius: 6px;
    padding: 4px 11px;
    margin: 4px 0 14px;
  }

  .demo-card h4 {
    color: #ffffff;
    font-size: 17px;
    font-weight: 700;
    line-height: 1.3;
    margin: 0 0 10px;
  }

  .demo-card p {
    color: #b0b0c2;
    font-size: 14px;
    line-height: 1.6;
    margin: 0 0 16px;
    flex-grow: 1;
  }

  
  .demo-card { background: #10101a; border: 1px solid #22222f; border-radius: 14px; padding: 20px; position: relative; overflow: hidden; display: flex; flex-direction: column; }
  .demo-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .demo-card.accent-blue::before   { background: linear-gradient(90deg, #4f7cff, #9b6bff); }
  .demo-card.accent-purple::before { background: linear-gradient(90deg, #9b6bff, #ff6bd6); }
  .demo-card.accent-cyan::before   { background: linear-gradient(90deg, #38bdf8, #4f7cff); }
  .demo-card.accent-green::before  { background: linear-gradient(90deg, #34d399, #38bdf8); }

  .demo-btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .demo-btn { font-size: 12px; font-weight: 500; padding: 6px 13px; border-radius: 7px; text-decoration: none; display: inline-flex; align-items: center; gap: 4px; }
  .demo-btn-primary { background: transparent; border: 1px solid #4f7cff; color: #8fa9ff; }
  .demo-btn-primary:hover { background: rgba(79, 124, 255, 0.12); }
  .demo-btn-secondary { background: transparent; border: 1px solid #2a2a3a; color: #b5b5c8; }
  .demo-btn-secondary:hover { background: #1a1a28; }
  .intro-line { color: #b0b0c2; font-size: 16px; line-height: 1.6; margin: 0 0 24px; }
  @media (max-width: 900px) { .demo-grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 560px) { .demo-grid { grid-template-columns: 1fr; } }
</style>

🔴 Key Achievements with Live Demos

<div class="demo-grid">
  <div class="demo-card accent-purple">
    <span class="demo-tag">LLM Fine-tuning</span>
    <h4>Small LLM for Ticket Triage</h4>
    <p>QLoRA fine-tuned Qwen3-4B for German/English ticket triage. Benchmarked on 25 real tickets against Claude Sonnet 4.6: core accuracy up 20% at ~17x lower cost ($0.29 vs $5.11 per 1,000 tickets).</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-secondary" href="YOUR_QWEN_GITHUB_LINK" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-blue">
    <span class="demo-tag">Agentic AI</span>
    <h4>Smart Dispatcher</h4>
    <p>LangGraph agent that triages support tickets and drafts dispatch plans, with human-in-the-loop approval, deterministic guardrails, FastMCP tools, and a self-correction loop.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://empire-smart-dispatcher-production.up.railway.app/" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/Smart-dispatcher-Agent" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-cyan">
    <span class="demo-tag">RAG</span>
    <h4>RAG Doc Chat</h4>
    <p>Multimodal RAG over uploaded PDFs: hybrid BGE + BM25 retrieval with rank fusion and reranking, streaming answers with page-level citations, and live eval metrics in the UI.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://raw.githubusercontent.com/devo002/RAG-doc-chat/main/asset/RAG.gif" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/RAG-doc-chat" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-blue">
    <span class="demo-tag">Computer Vision</span>
    <h4>Coil Winding Defect Classifier</h4>
    <p>Sample coil-winding images run through a trained multi-label classifier for defect detection — a deployment snippet from the full research project at FAPS Lab.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://raw.githubusercontent.com/devo002/devo002.github.io/master/assets/objectdetect.gif" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/Semi-supervised-Learning-for-Visual-Coil-winding-defect-detection" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-green">
    <span class="demo-tag">Full-stack</span>
    <h4>Budget Tracker</h4>
    <p>Finance tracker with weekly spending limits, real-time summaries, and category breakdowns.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://raw.githubusercontent.com/devo002/Budget-Tracker/main/assets/Animation.gif" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/Budget-Tracker" target="_blank">Code</a>
    </div>
  </div>
</div>


---


## Technical Skills {#skills}

<style>
  .skills-dark { max-width: 900px; margin: 24px 0; }
  .skills-category { margin: 0 0 26px; }
  .skills-category h4 { color: #ffffff; font-size: 16px; font-weight: 600; margin: 0 0 12px; }
  .skill-pill-row { display: flex; flex-wrap: wrap; gap: 10px; }
  .skill-pill {
    display: inline-block; background: #15151f; border: 1px solid #2a2a3a;
    color: #d6d6e0; font-size: 13px; padding: 7px 16px; border-radius: 999px;
    line-height: 1.2; white-space: nowrap;
  }
</style>

<div class="skills-dark">
  <div class="skills-category">
    <h4>Programming Languages</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">Python (Advanced)</span>
      <span class="skill-pill">JavaScript</span>
      <span class="skill-pill">TypeScript</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>ML &amp; Deep Learning</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">PyTorch</span>
      <span class="skill-pill">TensorFlow</span>
      <span class="skill-pill">Scikit-learn</span>
      <span class="skill-pill">NumPy</span>
      <span class="skill-pill">Pandas</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>Computer Vision</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">OpenCV</span>
      <span class="skill-pill">Image Processing</span>
      <span class="skill-pill">CNNs</span>
      <span class="skill-pill">Vision Transformers</span>
      <span class="skill-pill">OCR</span>
      <span class="skill-pill">GANs</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>LLMs &amp; AI Engineering</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">OpenAI</span>
      <span class="skill-pill">Anthropic</span>
      <span class="skill-pill">Gemini</span>
      <span class="skill-pill">Llama</span>
      <span class="skill-pill">Hugging Face</span>
      <span class="skill-pill">LangChain</span>
      <span class="skill-pill">LangGraph</span>
      <span class="skill-pill">LlamaIndex</span>
      <span class="skill-pill">CrewAI</span>
      <span class="skill-pill">RAG</span>
      <span class="skill-pill">QLoRA Fine-tuning</span>
      <span class="skill-pill">Tool Calling</span>
      <span class="skill-pill">MCP (FastMCP)</span>
      <span class="skill-pill">Structured Outputs (Pydantic)</span>
      <span class="skill-pill">Prompt Engineering</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>Web &amp; Full-Stack Development</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">React</span>
      <span class="skill-pill">Next.js</span>
      <span class="skill-pill">Node.js</span>
      <span class="skill-pill">REST APIs</span>
      <span class="skill-pill">Prisma ORM</span>
      <span class="skill-pill">Tailwind CSS</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>Databases &amp; Vector Databases</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">PostgreSQL</span>
      <span class="skill-pill">MySQL</span>
      <span class="skill-pill">MongoDB</span>
      <span class="skill-pill">Qdrant</span>
      <span class="skill-pill">Pinecone</span>
      <span class="skill-pill">ChromaDB</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>MLOps, Cloud &amp; Deployment</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">AWS</span>
      <span class="skill-pill">Docker</span>
      <span class="skill-pill">FastAPI</span>
      <span class="skill-pill">vLLM</span>
      <span class="skill-pill">Weights &amp; Biases</span>
      <span class="skill-pill">LangSmith</span>
      <span class="skill-pill">CI/CD</span>
      <span class="skill-pill">GitHub Actions</span>
    </div>
  </div>

  <div class="skills-category">
    <h4>Automation &amp; Workflows</h4>
    <div class="skill-pill-row">
      <span class="skill-pill">n8n</span>
      <span class="skill-pill">Make</span>
      <span class="skill-pill">AI Workflow Orchestration</span>
    </div>
  </div>
</div>


---

## Research Experience

<style>
  .research-timeline { position: relative; margin: 20px 0; }
  .research-item { display: grid; grid-template-columns: 150px 1fr; column-gap: 24px; position: relative; padding-bottom: 30px; }
  .research-item:last-child { padding-bottom: 0; }
  .research-item:not(:last-child)::before {
    content: ""; position: absolute; top: 28px; bottom: -30px; left: 162px; border-left: 1px dashed #33333f;
  }
  .research-date { color: #8a8aa0; font-size: 13px; padding-top: 20px; text-align: right; }
  .research-card { background: #10101a; border: 1px solid #22222f; border-radius: 14px; padding: 22px 24px; }
  .research-card h4 { color: #ffffff; font-size: 17px; font-weight: 700; margin: 0 0 4px; }
  .research-card .research-org { color: #8a8aa0; font-size: 13px; margin: 0 0 14px; }
  .research-card ul { margin: 0 0 16px; padding-left: 18px; color: #b5b5c8; font-size: 14px; line-height: 1.65; }
  .research-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .research-tag { background: #15151f; border: 1px solid #2a2a3a; color: #d6d6e0; font-size: 12px; padding: 5px 12px; border-radius: 999px; }
  .research-link { font-size: 13px; color: #8fa9ff; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }
  .research-link:hover { text-decoration: underline; }
</style>

<div class="research-timeline">

  <div class="research-item">
    <div class="research-date">Jun 2025 — Jan 2026</div>
    <div class="research-card">
      <h4>Machine Learning Researcher</h4>
      <p class="research-org">Pattern Recognition Lab, FAU</p>
      <ul>
        <li>Fine-tuned and evaluated alternative backbone architectures for the AFFGAN handwriting style encoder to better capture writer-specific features, benchmarking against the original baseline.</li>
        <li>Extended the system beyond word-level synthesis to line-level handwriting generation, adapting an existing line-generation architecture to support full-sentence output.</li>
        <li>Deployed the pipeline as a web application where users select a writing style, enter custom text, and instantly generate a handwriting image.</li>
        <li>Integrated TrOCR into a teacher–student distillation framework to evaluate and improve readability of generated handwriting, improving OCR accuracy by 20% over the baseline.</li>
      </ul>
      <div class="research-tags">
        <span class="research-tag">PyTorch</span>
        <span class="research-tag">GANs</span>
        <span class="research-tag">CNN</span>
        <span class="research-tag">ViT</span>
        <span class="research-tag">TrOCR</span>
        <span class="research-tag">Streamlit</span>
        <span class="research-tag">CUDA</span>
      </div>
      <a class="research-link" href="https://github.com/devo002/Handwriting_generation" target="_blank">View Code →</a>
    </div>
  </div>

  <div class="research-item">
    <div class="research-date">Apr 2025 — Sep 2025</div>
    <div class="research-card">
      <h4>Computer Vision Project</h4>
      <p class="research-org">Pattern Recognition Lab, FAU</p>
      <ul>
        <li>Built selective search object detection end-to-end and a video face recognition system combining MTCNN, FaceNet embeddings, and from-scratch k-NN / k-means.</li>
      </ul>
      <div class="research-tags">
        <span class="research-tag">Python</span>
        <span class="research-tag">OpenCV</span>
        <span class="research-tag">ResNet-50</span>
        <span class="research-tag">Scikit-learn</span>
      </div>
    </div>
  </div>

  <div class="research-item">
    <div class="research-date">Dec 2023 — Jan 2025</div>
    <div class="research-card">
      <h4>Computer Vision Researcher</h4>
      <p class="research-org">Institute for Factory Automation and Production Systems (FAPS)</p>
      <ul>
        <li>Collaborated in a research team on systematic data cleaning of thousands of industrial coil-winding images: manually corrected label noise on a representative subset, then used Cleanlab Studio's autofix to propagate corrections across the rest, establishing a high-fidelity baseline for training.</li>
        <li>Trained and evaluated fully-supervised baselines (DINOv2-L and EfficientNetV2-S) to set reference macro-F1, benchmarking against SSL (FixMatch / MixMatch) gains across four label-fraction splits (10%, 25%, 50%, 100%).</li>
        <li>Ran systematic hyperparameter optimization with Optuna across backbone/algorithm combinations (learning rate, batch size, augmentation), tracking all trials in a persistent SQLite results database for reproducibility.</li>
        <li>Designed and implemented a config-driven multi-label SSL training pipeline supporting FixMatch and MixMatch, reaching 90% macro-F1 with FixMatch.</li>
      </ul>
      <div class="research-tags">
        <span class="research-tag">PyTorch</span>
        <span class="research-tag">DINOv2-L</span>
        <span class="research-tag">EfficientNetV2-L</span>
        <span class="research-tag">FixMatch</span>
        <span class="research-tag">Optuna</span>
        <span class="research-tag">Cleanlab</span>
        <span class="research-tag">WandB</span>
        <span class="research-tag">Docker</span>
      </div>
      <a class="research-link" href="https://github.com/devo002/Semi-supervised-Learning-for-Visual-Coil-winding-defect-detection" target="_blank">View Code →</a>
    </div>
  </div>

</div>

---

## AI Engineering Projects {#projects}

<style>
  .ai-proj-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin: 20px 0; }
  .ai-proj-card { background: #10101a; border: 1px solid #22222f; border-radius: 14px; padding: 22px; position: relative; overflow: hidden; }
  .ai-proj-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .ai-proj-card.accent-blue::before   { background: linear-gradient(90deg, #4f7cff, #9b6bff); }
  .ai-proj-card.accent-purple::before { background: linear-gradient(90deg, #9b6bff, #ff6bd6); }
  .ai-proj-card.accent-cyan::before   { background: linear-gradient(90deg, #38bdf8, #4f7cff); }
  .ai-proj-card h4 { color: #ffffff; font-size: 18px; font-weight: 700; margin: 6px 0 6px; }
  .ai-proj-card .ai-proj-date { color: #7a7a90; font-size: 12px; margin: 0 0 12px; }
  .ai-proj-card p { color: #a8a8ba; font-size: 14px; line-height: 1.55; margin: 0 0 18px; }
  .ai-proj-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .ai-proj-tag { background: #15151f; border: 1px solid #2a2a3a; color: #d6d6e0; font-size: 12px; padding: 5px 12px; border-radius: 999px; }
  .ai-proj-link { font-size: 13px; color: #8fa9ff; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; }
  .ai-proj-link:hover { text-decoration: underline; }
</style>

<div class="ai-proj-grid">

  <div class="ai-proj-card accent-purple">
    <h4>Fine-tuning a Small LLM for Ticket Triage</h4>
    <p class="ai-proj-date">Qwen3-4B · QLoRA · 09/2026 – 10/2026</p>
    <p>Fine-tuned Qwen3-4B with QLoRA for German/English support-ticket triage. On 25 real tickets versus a production Claude Sonnet 4.6 call, core accuracy improved by 20% and cost fell ~17x ($0.29 vs $5.11 per 1,000 tickets); p50 latency was 2.95s vs 2.08s. Runs tracked in Weights &amp; Biases, merged model served with vLLM behind FastAPI and Docker.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">PyTorch</span>
      <span class="ai-proj-tag">Hugging Face</span>
      <span class="ai-proj-tag">QLoRA</span>
      <span class="ai-proj-tag">vLLM</span>
      <span class="ai-proj-tag">FastAPI</span>
      <span class="ai-proj-tag">Docker</span>
      <span class="ai-proj-tag">Weights &amp; Biases</span>
    </div>
    <a class="ai-proj-link" href="YOUR_QWEN_GITHUB_LINK" target="_blank">View Code →</a>
  </div>

  <div class="ai-proj-card accent-blue">
    <h4>Smart Dispatcher — Agentic Ticket Triage &amp; Dispatch</h4>
    <p class="ai-proj-date">04/2026 – 09/2026</p>
    <p>Multi-step LangGraph agent that triages support tickets and drafts dispatch plans, with a human-in-the-loop checkpoint where a reviewer approves or edits every plan. Regex-based guardrails validate LLM output, inventory/manuals/technician lookup are exposed as FastMCP tools, and a self-correction loop retrieves firmware workarounds via RAG (100% recovery on failures in testing). Backed by a 17-case eval suite and a React/FastAPI dashboard that streams each step over SSE, traced in LangSmith.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">LangGraph</span>
      <span class="ai-proj-tag">FastMCP</span>
      <span class="ai-proj-tag">ChromaDB</span>
      <span class="ai-proj-tag">FastAPI</span>
      <span class="ai-proj-tag">React</span>
      <span class="ai-proj-tag">LangSmith</span>
      <span class="ai-proj-tag">Docker</span>
      <span class="ai-proj-tag">Railway</span>
    </div>
    <a class="ai-proj-link" href="https://empire-smart-dispatcher-production.up.railway.app/" target="_blank">Live Demo →</a>
    <a class="ai-proj-link" href="https://github.com/devo002/Smart-dispatcher-Agent" target="_blank">View Code →</a>
  </div>

  <div class="ai-proj-card accent-cyan">
    <h4>RAG-Powered Document Chat System</h4>
    <p class="ai-proj-date">02/2026 – 06/2026</p>
    <p>Publicly deployed multimodal RAG app: upload PDFs and get streaming, grounded answers with page-level citations. Hybrid retrieval combines dense BGE embeddings and BM25 with Reciprocal Rank Fusion and LLM reranking across text and image chunks, with live Hit Rate, MRR, faithfulness, answer relevancy and correctness shown in the UI.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">FastAPI</span>
      <span class="ai-proj-tag">LlamaIndex</span>
      <span class="ai-proj-tag">Qdrant</span>
      <span class="ai-proj-tag">SSE</span>
      <span class="ai-proj-tag">Docker</span>
      <span class="ai-proj-tag">Railway</span>
      <span class="ai-proj-tag">GitHub CI/CD</span>
    </div>
    <a class="ai-proj-link" href="https://raw.githubusercontent.com/devo002/RAG-doc-chat/main/asset/RAG.gif" target="_blank">Demo →</a>
    <a class="ai-proj-link" href="https://github.com/devo002/RAG-doc-chat" target="_blank">View Code →</a>
  </div>

  <div class="ai-proj-card accent-blue">
    <h4>AI Ticket Bridge Agent</h4>
    <p class="ai-proj-date">Teams → Jira Automation · 02/2026 – 04/2026</p>
    <p>AI bridge agent that classifies Microsoft Teams conversations, enriches them with CRM context, and reduces manual ticket creation to a one-click approval workflow. Processes 70+ support messages in fewer than four Claude API calls while streaming progress to a review dashboard, where humans validate recommendations before approved actions sync to Jira and Teams.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">Flask</span>
      <span class="ai-proj-tag">JavaScript</span>
      <span class="ai-proj-tag">Claude</span>
      <span class="ai-proj-tag">Jira REST API</span>
      <span class="ai-proj-tag">Docker</span>
    </div>
    <a class="ai-proj-link" href="https://github.com/devo002/AI_Ticket_Agent" target="_blank">View Code →</a>
  </div>

  <div class="ai-proj-card accent-purple">
    <h4>AI-Powered Job Matching &amp; Application Assistant</h4>
    <p class="ai-proj-date">01/2026 – 02/2026</p>
    <p>n8n pipeline that monitors job emails, embeds descriptions, and scores them against a CV using Pinecone semantic search — classifying each as Apply / Maybe / Skip.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">n8n</span>
      <span class="ai-proj-tag">OpenAI</span>
      <span class="ai-proj-tag">Pinecone</span>
      <span class="ai-proj-tag">RAG</span>
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">IMAP</span>
    </div>
  </div>

  <div class="ai-proj-card accent-cyan">
    <h4>Log Analysis System using LLMs on AWS</h4>
    <p class="ai-proj-date">12/2025 – 01/2026</p>
    <p>Event-driven pipeline on AWS Bedrock + Lambda that auto-clusters logs, infers root causes, and persists results — replacing manual debugging.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">AWS</span>
      <span class="ai-proj-tag">Bedrock</span>
      <span class="ai-proj-tag">Lambda</span>
      <span class="ai-proj-tag">S3</span>
      <span class="ai-proj-tag">CloudWatch</span>
      <span class="ai-proj-tag">FastAPI</span>
    </div>
    <a class="ai-proj-link" href="https://github.com/devo002/LLM-log-analyzer" target="_blank">View Code →</a>
  </div>

  <div class="ai-proj-card accent-blue">
    <h4>RAG Chatbot with Nvidia</h4>
    <p class="ai-proj-date">10/2025 – 11/2025</p>
    <p>Document-grounded chatbot with persistent conversation history across PDF and HTML uploads, containerised with Docker.</p>
    <div class="ai-proj-tags">
      <span class="ai-proj-tag">Python</span>
      <span class="ai-proj-tag">NVIDIA</span>
      <span class="ai-proj-tag">Llama</span>
      <span class="ai-proj-tag">FAISS</span>
      <span class="ai-proj-tag">FastAPI</span>
      <span class="ai-proj-tag">Docker</span>
    </div>
    <a class="ai-proj-link" href="https://github.com/devo002/RAG-chatbot-nvidia" target="_blank">View Code →</a>
  </div>

</div>


---

## Education
### Master of Science (M.Sc.) in Artificial Intelligence
**Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**, Germany  
*2022 – 2026*

### Bachelor of Technology in Computer Science
**Federal University of Technology Minna**, Nigeria  
*2014 – 2018*

## Languages
English (Native) — German (Intermediate)

---

Tools & Libraries

<style>
  .tools-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
  .tools-chip {
    width: 56px; height: 56px;
    display: flex; align-items: center; justify-content: center;
    background: #10101a; border: 1px solid #22222f; border-radius: 12px;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }
  .tools-chip:hover { border-color: #4f7cff; transform: translateY(-2px); }
  .tools-chip img { width: 26px; height: 26px; filter: invert(1) brightness(1.7); opacity: 0.85; }
  .tools-chip:hover img { opacity: 1; }
</style>

<div class="tools-grid">
  <div class="tools-chip"><img src="/assets/icons/pytorch.svg" alt="Pytorch" /></div>
  <div class="tools-chip"><img src="/assets/icons/tensorflow.svg" alt="Tensorflow" /></div>
  <div class="tools-chip"><img src="/assets/icons/opencv.svg" alt="Open CV" /></div>
  <div class="tools-chip"><img src="/assets/icons/jupyter.svg" alt="Jupyter" /></div>
  <div class="tools-chip"><img src="/assets/icons/python.svg" alt="Python" /></div>
  <div class="tools-chip"><img src="/assets/icons/javascript.svg" alt="Javascript" /></div>
  <div class="tools-chip"><img src="/assets/icons/nodedotjs.svg" alt="NodeJS" /></div>
  <div class="tools-chip"><img src="/assets/icons/mongodb.svg" alt="MongoDB" /></div>
  <div class="tools-chip"><img src="/assets/icons/huggingface.svg" alt="Hugging Face" /></div>
  <div class="tools-chip"><img src="/assets/icons/git.svg" alt="Git" /></div>
  <div class="tools-chip"><img src="/assets/icons/github.svg" alt="Github" /></div>
  <div class="tools-chip"><img src="/assets/icons/numpy.svg" alt="Numpy" /></div>
  <div class="tools-chip"><img src="/assets/icons/docker.svg" alt="Docker" /></div>
  <div class="tools-chip"><img src="/assets/icons/langchain.svg" alt="Langchain" /></div>
  <div class="tools-chip"><img src="/assets/icons/optuna.svg" alt="Optuna" /></div>
</div>
---
## Contact

<style>
  .contact-wrap { margin: 24px 0; }
  .contact-wrap h2 { color: #ffffff; font-size: 24px; font-weight: 700; margin: 0 0 10px; }
  .contact-blurb { color: #a8a8ba; font-size: 14px; line-height: 1.6; margin: 0 0 20px; }
  .contact-card {
    display: inline-flex; align-items: center; gap: 14px;
    background: #10101a; border: 1px solid #22222f; border-radius: 12px;
    padding: 14px 20px; text-decoration: none; transition: border-color 0.15s ease;
  }
  .contact-card:hover { border-color: #4f7cff; }
  .contact-icon {
    width: 36px; height: 36px; border-radius: 8px; background: #2a1f3d;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .contact-icon svg { width: 17px; height: 17px; fill: #c9a8ff; }
  .contact-label { color: #8a8aa0; font-size: 12px; margin: 0 0 2px; }
  .contact-value { color: #f0f0f5; font-size: 14px; font-weight: 600; }
</style>

<div class="contact-wrap">
  <p class="contact-blurb">I'm open to AI/ML engineering, computer vision, and applied GenAI roles — reach out anytime.</p>

  <a class="contact-card" href="mailto:davidmayowaonaiyekan@gmail.com">
    <div class="contact-icon">
      <svg viewBox="0 0 24 24"><path d="M2 4h20v16H2V4zm2 2v.01L12 12l8-5.99V6H4zm16 2.24l-7.4 5.55a1 1 0 0 1-1.2 0L4 8.24V18h16V8.24z"/></svg>
    </div>
    <div>
      <div class="contact-label">Email</div>
      <div class="contact-value">davidmayowaonaiyekan@gmail.com</div>
    </div>
  </a>
</div>
