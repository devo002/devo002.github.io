---
layout: single
title: ""
permalink: /
author_profile: true   # hide the big name card on the left # grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
---


<style>
  .demo-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin: 16px 0 24px; }
  .demo-card { background: #10101a; border: 1px solid #22222f; border-radius: 14px; padding: 20px; position: relative; overflow: hidden; display: flex; flex-direction: column; }
  .demo-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
  .demo-card.accent-blue::before   { background: linear-gradient(90deg, #4f7cff, #9b6bff); }
  .demo-card.accent-purple::before { background: linear-gradient(90deg, #9b6bff, #ff6bd6); }
  .demo-card.accent-cyan::before   { background: linear-gradient(90deg, #38bdf8, #4f7cff); }
  .demo-card.accent-green::before  { background: linear-gradient(90deg, #34d399, #38bdf8); }
  .demo-tag { align-self: flex-start; font-size: 11px; font-weight: 600; letter-spacing: 0.02em; background: #1a1a28; border: 1px solid #2a2a3a; color: #9fb4ff; border-radius: 5px; padding: 3px 9px; margin: 6px 0 12px; }
  .demo-card h4 { color: #ffffff; font-size: 16px; font-weight: 700; margin: 0 0 10px; }
  .demo-card p { color: #a8a8ba; font-size: 13px; line-height: 1.55; margin: 0 0 16px; flex-grow: 1; }
  .demo-btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .demo-btn { font-size: 12px; font-weight: 500; padding: 6px 13px; border-radius: 7px; text-decoration: none; display: inline-flex; align-items: center; gap: 4px; }
  .demo-btn-primary { background: transparent; border: 1px solid #4f7cff; color: #8fa9ff; }
  .demo-btn-primary:hover { background: rgba(79, 124, 255, 0.12); }
  .demo-btn-secondary { background: transparent; border: 1px solid #2a2a3a; color: #b5b5c8; }
  .demo-btn-secondary:hover { background: #1a1a28; }
  @media (max-width: 900px) { .demo-grid { grid-template-columns: repeat(2, 1fr); } }
  @media (max-width: 560px) { .demo-grid { grid-template-columns: 1fr; } }
</style>

🔴 Key Achievements with Live Demos

<div class="demo-grid">
  <div class="demo-card accent-blue">
    <span class="demo-tag">Computer Vision</span>
    <h4>Coil Winding Defect Classifier</h4>
    <p>Sample coil-winding images run through a trained multi-label classifier for defect detection — a deployment snippet from the full research project at FAPS Lab.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="http://98.89.229.163:8000/" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/Semi-supervised-Learning-for-Visual-Coil-winding-defect-detection" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-purple">
    <span class="demo-tag">Agentic AI</span>
    <h4>Smart Dispatcher</h4>
    <p>LangGraph agent that triages support tickets with a 6-node self-correction loop.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://empire-smart-dispatcher-production.up.railway.app/" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/Smart-dispatcher-Agent" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-cyan">
    <span class="demo-tag">RAG</span>
    <h4>RAG Doc Chat</h4>
    <p>Upload PDFs and chat with them — streaming answers with inline source citations and an automated RAGAS eval pipeline.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://exemplary-manifestation-production-dcca.up.railway.app/" target="_blank">▶ Open demo</a>
      <a class="demo-btn demo-btn-secondary" href="https://github.com/devo002/RAG-doc-chat" target="_blank">Code</a>
    </div>
  </div>

  <div class="demo-card accent-green">
    <span class="demo-tag">Full-stack</span>
    <h4>Budget Tracker</h4>
    <p>Finance tracker with weekly spending limits, real-time summaries, and category breakdowns.</p>
    <div class="demo-btn-row">
      <a class="demo-btn demo-btn-primary" href="https://budget-tracker-lac-six.vercel.app/dashboard" target="_blank">▶ Open demo</a>
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
      <span class="skill-pill">Python</span>
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
      <span class="skill-pill">Llama</span>
      <span class="skill-pill">Hugging Face</span>
      <span class="skill-pill">LangChain</span>
      <span class="skill-pill">LangGraph</span>
      <span class="skill-pill">LlamaIndex</span>
      <span class="skill-pill">RAG</span>
      <span class="skill-pill">Prompt Engineering</span>
      <span class="skill-pill">MCP</span>
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
    <div class="research-date">Jun 2025 — Nov 2025</div>
    <div class="research-card">
      <h4>Machine Learning Researcher</h4>
      <p class="research-org">Pattern Recognition Lab, FAU</p>
      <ul>
        <li>Integrated a custom Transformer-based encoder into the AFFGANwriting pipeline, boosting perceptual pick-rates by 40% and OCR accuracy by 20% via a teacher-student alignment framework along the TrOCR model.</li>
      </ul>
      <div class="research-tags">
        <span class="research-tag">PyTorch</span>
        <span class="research-tag">GANs</span>
        <span class="research-tag">TrOCR</span>
        <span class="research-tag">Streamlit</span>
        <span class="research-tag">CUDA</span>
      </div>
      <a class="research-link" href="https://github.com/devo002/Handwriting_generation" target="_blank">View Code →</a>
    </div>
  </div>

  <div class="research-item">
    <div class="research-date">Dec 2023 — Dec 2024</div>
    <div class="research-card">
      <h4>Computer Vision Researcher</h4>
      <p class="research-org">Institute for Factory Automation and Production Systems (FAPS)</p>
      <ul>
        <li>Designed and implemented a semi-supervised learning pipeline using FixMatch + DINOv2-L SSL on unlabelled coil-winding images, reaching 90% macro F1 on multi-label defect classification (vs. 87% supervised baseline).</li>
      </ul>
      <div class="research-tags">
        <span class="research-tag">PyTorch</span>
        <span class="research-tag">DINOv2-L</span>
        <span class="research-tag">FixMatch</span>
        <span class="research-tag">Optuna</span>
        <span class="research-tag">Tensorboard</span>
      </div>
      <a class="research-link" href="https://github.com/devo002/Semi-supervised-Learning-for-Visual-Coil-winding-defect-detection" target="_blank">View Code →</a>
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
  .ai-proj-link { font-size: 13px; color: #8fa9ff; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }
  .ai-proj-link:hover { text-decoration: underline; }
</style>

<div class="ai-proj-grid">

  <div class="ai-proj-card accent-blue">
    <h4>AI Ticket Bridge Agent</h4>
    <p class="ai-proj-date">Teams → Jira Automation · 04/2026</p>
    <p>Multi-step agent that classifies 70+ Microsoft Teams messages in under 4 Claude API calls, cross-references HubSpot ARR, deduplicates against Jira, and streams decisions to a full-stack review UI.</p>
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
*2022 – 2025*

### Bachelor of Technology in Computer Science
**Federal University of Technology Minna**, Nigeria  
*2014 – 2018*

---

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

---

📧 davidmayowaonaiyekan@gmail.com