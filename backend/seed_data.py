"""Seed script to populate ChromaDB with realistic job board data.

Uses real Gemini embeddings for high-quality semantic search results.

Run with: python seed_data.py

Requires:
- GOOGLE_API_KEY set in .env
- ChromaDB running (chroma run --port 8001)
"""

import asyncio
import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import chromadb


# =============================================================================
# SAMPLE DATA — Realistic job board entries
# =============================================================================

SAMPLE_JOB_POSTS = [
    # Python / Backend
    {
        "id": "job_1",
        "title": "Senior Python Developer",
        "description": "We are looking for a Senior Python Developer to join our backend team. You will design and build scalable microservices, mentor junior developers, and work closely with the DevOps team to deploy services on AWS. Our stack includes FastAPI, PostgreSQL, Redis, and Celery for async task processing.",
        "requirements": "5+ years Python experience, FastAPI or Django, PostgreSQL, Docker, AWS, CI/CD pipelines. Strong problem-solving skills and experience with distributed systems.",
        "location": "Remote",
        "salary_range": "$130,000 - $160,000",
        "company": "TechNova Solutions",
    },
    {
        "id": "job_2",
        "title": "Junior Python Backend Developer",
        "description": "Join our growing engineering team as a Junior Python Developer. You will work on REST API development, write unit tests, and learn from senior engineers. Great opportunity for freshers who want to build production systems from day one.",
        "requirements": "1+ years Python, basic knowledge of Flask or FastAPI, SQL fundamentals, Git. Computer Science degree or equivalent bootcamp. Eagerness to learn.",
        "location": "Remote",
        "salary_range": "$60,000 - $80,000",
        "company": "DataStream Analytics",
    },
    {
        "id": "job_3",
        "title": "Python Django Developer",
        "description": "We need a mid-level Django developer to maintain and extend our e-commerce platform. You will build new features, optimize database queries, integrate payment gateways (Stripe, PayPal), and write comprehensive tests.",
        "requirements": "3+ years Django, PostgreSQL, REST APIs, Celery, Redis. Experience with e-commerce systems and payment integrations preferred.",
        "location": "New York, NY (Hybrid)",
        "salary_range": "$100,000 - $130,000",
        "company": "ShopFlow Inc.",
    },
    # Frontend
    {
        "id": "job_4",
        "title": "Frontend React Engineer",
        "description": "Join our product team to build beautiful, responsive web applications using React and TypeScript. You will collaborate with designers and backend engineers to deliver delightful user experiences. We use Next.js for SSR and Tailwind CSS for styling.",
        "requirements": "3+ years React/TypeScript, Next.js, Tailwind CSS, REST APIs, Git. Experience with design systems and accessibility standards is a plus.",
        "location": "San Francisco, CA",
        "salary_range": "$120,000 - $150,000",
        "company": "PixelCraft Design Studio",
    },
    {
        "id": "job_5",
        "title": "Senior Frontend Developer (Vue.js)",
        "description": "Lead frontend development for our fintech dashboard. Build complex data visualizations, real-time trading interfaces, and ensure sub-100ms render times. Mentor junior frontend developers and establish coding standards.",
        "requirements": "5+ years frontend, 3+ years Vue.js/Nuxt.js, TypeScript, D3.js or Chart.js, WebSocket experience, performance optimization. Fintech experience preferred.",
        "location": "London, UK (Remote OK)",
        "salary_range": "£80,000 - £110,000",
        "company": "DataStream Analytics",
    },
    # Data Science / ML
    {
        "id": "job_6",
        "title": "Data Scientist - Machine Learning",
        "description": "We need a Data Scientist to develop and deploy ML models for our recommendation engine. You will work with large datasets (50M+ records), build predictive models using deep learning, and integrate them into production systems via MLflow.",
        "requirements": "MS/PhD in CS or Statistics, Python, scikit-learn, TensorFlow/PyTorch, SQL, A/B testing, MLOps experience preferred. Published research is a bonus.",
        "location": "Remote",
        "salary_range": "$140,000 - $180,000",
        "company": "TechNova Solutions",
    },
    {
        "id": "job_7",
        "title": "Junior Data Analyst",
        "description": "Start your data career with us! Analyze business metrics, create dashboards in Looker/Tableau, write SQL queries to extract insights, and present findings to stakeholders. Perfect for recent graduates who love working with data.",
        "requirements": "Bachelor's in Statistics, Mathematics, or CS. SQL proficiency, Excel/Google Sheets, basic Python or R. Tableau or Looker experience is a plus. Strong communication skills.",
        "location": "Austin, TX",
        "salary_range": "$55,000 - $75,000",
        "company": "GreenLeaf HealthTech",
    },
    {
        "id": "job_8",
        "title": "NLP Engineer",
        "description": "Build and deploy natural language processing models for our customer support automation platform. Work on intent classification, named entity recognition, sentiment analysis, and conversational AI using transformer-based architectures.",
        "requirements": "3+ years NLP/ML, Python, Hugging Face Transformers, PyTorch, BERT/GPT fine-tuning, Docker, AWS SageMaker. Experience with production NLP systems required.",
        "location": "Remote",
        "salary_range": "$150,000 - $190,000",
        "company": "TechNova Solutions",
    },
    # DevOps / Cloud
    {
        "id": "job_9",
        "title": "DevOps Engineer",
        "description": "We are hiring a DevOps Engineer to manage our cloud infrastructure on AWS. You will automate deployments using Terraform, manage Kubernetes clusters (EKS), set up monitoring with Datadog, and ensure 99.9% uptime for our microservices.",
        "requirements": "3+ years DevOps, AWS (EC2, ECS, Lambda, RDS), Kubernetes, Terraform, Docker, CI/CD (GitHub Actions), monitoring tools (Datadog/Prometheus). On-call rotation required.",
        "location": "Berlin, Germany",
        "salary_range": "€70,000 - €95,000",
        "company": "CloudPeak Infrastructure",
    },
    {
        "id": "job_10",
        "title": "Cloud Solutions Architect",
        "description": "Design and implement cloud-native architectures for enterprise clients. Lead technical discussions, create architecture diagrams, conduct proof-of-concept implementations, and guide development teams. Travel to client sites 2-3 days per month.",
        "requirements": "7+ years software engineering, 3+ years cloud architecture, AWS/Azure/GCP certifications (at least one), microservices, event-driven architecture, strong presentation skills.",
        "location": "Chicago, IL (Hybrid)",
        "salary_range": "$160,000 - $200,000",
        "company": "CloudPeak Infrastructure",
    },
    {
        "id": "job_11",
        "title": "Site Reliability Engineer (SRE)",
        "description": "Join our SRE team to keep our platform running at scale (10M daily active users). You will design fault-tolerant systems, automate incident response, conduct capacity planning, and lead post-incident reviews. On-call rotation with generous compensation.",
        "requirements": "4+ years SRE or DevOps, Linux systems administration, Python/Go scripting, Kubernetes, observability tools (Grafana, PagerDuty), chaos engineering experience. SLO/SLI/error budget methodology.",
        "location": "Remote (US timezone)",
        "salary_range": "$145,000 - $185,000",
        "company": "TechNova Solutions",
    },
    # Full Stack
    {
        "id": "job_12",
        "title": "Full Stack Developer (Node.js + React)",
        "description": "Build end-to-end features for our SaaS platform serving 50K+ businesses. Work across the stack from React frontend to Node.js/Express backend with PostgreSQL database. Ship features every two weeks in an agile environment.",
        "requirements": "3+ years full stack, Node.js, Express, React, TypeScript, PostgreSQL, REST APIs, Git, basic AWS knowledge. Experience with SaaS products preferred.",
        "location": "Toronto, Canada (Hybrid)",
        "salary_range": "CAD $95,000 - $125,000",
        "company": "ShopFlow Inc.",
    },
    {
        "id": "job_13",
        "title": "Full Stack Python Developer",
        "description": "Work on our healthcare platform from frontend to backend. Build patient-facing React dashboards, develop FastAPI microservices, manage PostgreSQL databases, and integrate with FHIR healthcare APIs. HIPAA compliance experience valued.",
        "requirements": "3+ years full stack, Python (FastAPI/Django), React/TypeScript, PostgreSQL, Docker. Healthcare/HIPAA experience is a strong plus. Security-conscious development practices.",
        "location": "Boston, MA",
        "salary_range": "$110,000 - $145,000",
        "company": "GreenLeaf HealthTech",
    },
    # Mobile
    {
        "id": "job_14",
        "title": "Mobile Developer - iOS (Swift)",
        "description": "Develop and maintain our iOS application used by 500K+ users. Implement new features using SwiftUI, optimize app performance, handle App Store submissions, and ensure accessibility compliance.",
        "requirements": "4+ years iOS development, Swift, UIKit/SwiftUI, Core Data, RESTful APIs, XCTest, CI/CD for mobile (Fastlane). App Store publishing experience required.",
        "location": "Seattle, WA",
        "salary_range": "$130,000 - $160,000",
        "company": "PixelCraft Design Studio",
    },
    {
        "id": "job_15",
        "title": "Android Developer (Kotlin)",
        "description": "Build and maintain our Android app for our telemedicine platform. Work with Jetpack Compose, implement real-time video calling features, integrate health device APIs (Bluetooth), and ensure HIPAA-compliant data handling.",
        "requirements": "3+ years Android development, Kotlin, Jetpack Compose, MVVM architecture, Retrofit, Room database, unit testing. Healthcare app experience preferred.",
        "location": "Remote",
        "salary_range": "$115,000 - $145,000",
        "company": "GreenLeaf HealthTech",
    },
    # Product / Management
    {
        "id": "job_16",
        "title": "Product Manager - AI Products",
        "description": "Lead the product strategy for our AI-powered recruitment features. Define roadmaps, prioritize features based on data, work with engineering and data science teams, conduct user interviews, and own the P&L for your product area.",
        "requirements": "5+ years product management, experience shipping AI/ML products, strong analytical skills (SQL, data analysis), excellent communication, Agile/Scrum methodology. MBA preferred but not required.",
        "location": "San Francisco, CA",
        "salary_range": "$150,000 - $190,000",
        "company": "TechNova Solutions",
    },
    {
        "id": "job_17",
        "title": "Engineering Manager - Backend",
        "description": "Manage a team of 8 backend engineers building our core platform. You will conduct 1:1s, drive technical strategy, remove blockers, hire and mentor engineers, and ensure delivery timelines are met while maintaining code quality.",
        "requirements": "5+ years engineering experience, 2+ years people management, strong Python/Java background, experience scaling teams, agile methodology. Track record of shipping products and growing engineers.",
        "location": "New York, NY",
        "salary_range": "$170,000 - $210,000",
        "company": "TechNova Solutions",
    },
    # QA / Testing
    {
        "id": "job_18",
        "title": "QA Automation Engineer",
        "description": "Build and maintain our automated testing framework covering web, API, and mobile platforms. Write Cypress E2E tests, Playwright visual tests, k6 performance tests, and integrate everything into our CI/CD pipeline.",
        "requirements": "3+ years QA automation, Cypress or Playwright, Python or JavaScript, API testing (Postman/Newman), CI/CD integration, performance testing (k6/JMeter). ISTQB certification is a plus.",
        "location": "Remote",
        "salary_range": "$90,000 - $120,000",
        "company": "ShopFlow Inc.",
    },
    # Design
    {
        "id": "job_19",
        "title": "Senior UI/UX Designer",
        "description": "Design intuitive interfaces for our B2B analytics platform. Conduct user research, create wireframes and high-fidelity prototypes in Figma, build and maintain our design system, and work closely with engineers to ship pixel-perfect designs.",
        "requirements": "5+ years UI/UX design, Figma expert, design systems experience, user research methods, responsive design, basic HTML/CSS knowledge, portfolio required. B2B/enterprise product experience preferred.",
        "location": "Remote",
        "salary_range": "$120,000 - $150,000",
        "company": "DataStream Analytics",
    },
    # Security
    {
        "id": "job_20",
        "title": "Application Security Engineer",
        "description": "Protect our platform and customer data. Conduct code reviews for security vulnerabilities, perform penetration testing, implement SAST/DAST tooling, lead security awareness training, and respond to security incidents.",
        "requirements": "4+ years application security, OWASP Top 10 expertise, Python/Go, experience with security tools (Burp Suite, Snyk, SonarQube), cloud security (AWS), CISSP/CEH certification preferred.",
        "location": "Remote (US)",
        "salary_range": "$140,000 - $175,000",
        "company": "TechNova Solutions",
    },
    # Data Engineering
    {
        "id": "job_21",
        "title": "Senior Data Engineer",
        "description": "Build and maintain our data platform processing 500GB+ daily. Design ETL pipelines using Apache Airflow, manage data warehouses (Snowflake/BigQuery), ensure data quality, and enable self-serve analytics for business teams.",
        "requirements": "5+ years data engineering, Python, SQL, Apache Airflow, Spark, Snowflake or BigQuery, dbt, Docker. Experience with real-time streaming (Kafka) preferred.",
        "location": "Remote",
        "salary_range": "$145,000 - $180,000",
        "company": "DataStream Analytics",
    },
    # Blockchain / Web3
    {
        "id": "job_22",
        "title": "Blockchain Developer (Solidity)",
        "description": "Develop smart contracts for our decentralized finance (DeFi) platform. Write and audit Solidity code, build integration tests using Hardhat, implement upgradeable contract patterns, and collaborate with our frontend team on Web3 integration.",
        "requirements": "2+ years Solidity development, Ethereum/EVM chains, Hardhat/Foundry, OpenZeppelin, DeFi protocols understanding, JavaScript/TypeScript, security audit experience preferred.",
        "location": "Remote (Global)",
        "salary_range": "$130,000 - $180,000",
        "company": "CryptoVault Labs",
    },
    # AI / LLM specific
    {
        "id": "job_23",
        "title": "AI/ML Engineer - LLM Applications",
        "description": "Build production LLM-powered features for our recruitment platform. Implement RAG pipelines, fine-tune models, build evaluation frameworks, optimize inference costs, and integrate with our existing microservices architecture.",
        "requirements": "3+ years ML engineering, Python, LangChain/LlamaIndex, OpenAI API, vector databases (Pinecone/ChromaDB/Weaviate), FastAPI, Docker. Experience deploying LLM applications to production.",
        "location": "Remote",
        "salary_range": "$155,000 - $200,000",
        "company": "TechNova Solutions",
    },
    # Internship
    {
        "id": "job_24",
        "title": "Software Engineering Intern (Summer 2025)",
        "description": "12-week paid internship for students passionate about building software. Work on real features alongside senior engineers, attend tech talks, receive mentorship, and present your project at the end of the internship. Previous interns have received full-time offers.",
        "requirements": "Currently pursuing BS/MS in Computer Science or related field. Knowledge of at least one programming language (Python, Java, JavaScript). Familiarity with Git. Strong problem-solving skills demonstrated through coursework or side projects.",
        "location": "San Francisco, CA (On-site)",
        "salary_range": "$45/hour",
        "company": "TechNova Solutions",
    },
    # Cybersecurity
    {
        "id": "job_25",
        "title": "Cloud Security Analyst",
        "description": "Monitor and secure our AWS cloud infrastructure. Configure GuardDuty, Security Hub, and CloudTrail. Perform threat modeling, implement least-privilege IAM policies, automate security compliance checks, and respond to alerts.",
        "requirements": "3+ years cloud security, AWS (IAM, VPC, Security Hub, GuardDuty), scripting (Python/Bash), knowledge of CIS benchmarks, SOC2 compliance experience. AWS Security Specialty certification preferred.",
        "location": "Washington, DC (Hybrid)",
        "salary_range": "$120,000 - $150,000",
        "company": "CloudPeak Infrastructure",
    },
]

SAMPLE_COMPANIES = [
    {
        "id": "comp_1",
        "name": "TechNova Solutions",
        "description": "TechNova Solutions is a fast-growing SaaS company building AI-powered tools for HR and recruitment. Founded in 2018, we have 200+ employees across offices in San Francisco, New York, and London. We serve Fortune 500 clients including Google, Microsoft, and Amazon. Our platform processes over 10 million job applications monthly using advanced NLP and matching algorithms.",
        "industry": "Enterprise SaaS / HR Tech",
        "founded": "2018",
        "size": "200-500 employees",
        "funding": "Series C ($85M)",
    },
    {
        "id": "comp_2",
        "name": "DataStream Analytics",
        "description": "DataStream Analytics specializes in big data processing and real-time analytics platforms. We help e-commerce and fintech companies make data-driven decisions at scale. Our proprietary streaming engine processes 2 billion events per day with sub-second latency. Founded by ex-Google engineers, we are backed by top-tier VCs.",
        "industry": "Data Analytics / FinTech",
        "founded": "2019",
        "size": "100-200 employees",
        "funding": "Series B ($42M)",
    },
    {
        "id": "comp_3",
        "name": "CloudPeak Infrastructure",
        "description": "CloudPeak Infrastructure provides managed cloud services and DevOps consulting to startups and enterprises. We have helped 300+ companies migrate to the cloud, reducing their infrastructure costs by an average of 40%. Our team of certified AWS/GCP/Azure architects delivers production-ready solutions in weeks, not months.",
        "industry": "Cloud Computing / Consulting",
        "founded": "2017",
        "size": "150-300 employees",
        "funding": "Profitable (No external funding)",
    },
    {
        "id": "comp_4",
        "name": "PixelCraft Design Studio",
        "description": "PixelCraft is a digital design and development agency focused on creating beautiful mobile and web experiences. We work with funded startups to build their MVPs and scale their products to millions of users. Our portfolio includes apps featured on the App Store and Google Play with combined 5M+ downloads.",
        "industry": "Design Agency / Mobile Development",
        "founded": "2020",
        "size": "50-100 employees",
        "funding": "Bootstrapped",
    },
    {
        "id": "comp_5",
        "name": "GreenLeaf HealthTech",
        "description": "GreenLeaf HealthTech builds telemedicine and health monitoring platforms used by 200+ hospitals and clinics across the US. Our platform enables remote patient monitoring, video consultations, and AI-powered symptom triage. We are on a mission to make healthcare accessible to everyone, regardless of location.",
        "industry": "HealthTech / Telemedicine",
        "founded": "2019",
        "size": "100-200 employees",
        "funding": "Series B ($55M)",
    },
    {
        "id": "comp_6",
        "name": "ShopFlow Inc.",
        "description": "ShopFlow is a headless commerce platform powering 50,000+ online stores worldwide. Our API-first architecture enables brands to build custom shopping experiences across web, mobile, and in-store. We process $2B+ in GMV annually and integrate with 100+ payment providers, shipping carriers, and marketing tools.",
        "industry": "E-Commerce / SaaS",
        "founded": "2016",
        "size": "300-500 employees",
        "funding": "Series D ($120M)",
    },
    {
        "id": "comp_7",
        "name": "CryptoVault Labs",
        "description": "CryptoVault Labs builds secure DeFi infrastructure for institutional investors. Our smart contract platform handles $500M+ in total value locked (TVL) with zero security incidents. We are a remote-first team of 40 engineers and researchers passionate about making decentralized finance safe and accessible.",
        "industry": "Blockchain / DeFi",
        "founded": "2021",
        "size": "30-50 employees",
        "funding": "Series A ($25M)",
    },
    {
        "id": "comp_8",
        "name": "EduSpark Learning",
        "description": "EduSpark Learning is an EdTech company building the next generation of online learning experiences. Our adaptive learning platform uses AI to personalize course content for each student. We partner with 50+ universities and have 2M+ active learners globally across STEM subjects.",
        "industry": "EdTech / AI Learning",
        "founded": "2020",
        "size": "80-120 employees",
        "funding": "Series A ($18M)",
    },
]

SAMPLE_CANDIDATES = [
    # Backend / Python
    {
        "id": "cand_1",
        "name": "Alex Chen",
        "skills": "Python, FastAPI, Django, PostgreSQL, Docker, AWS, Redis, Celery, Kubernetes, GraphQL",
        "experience": "6 years as a backend developer. Built microservices handling 10M+ requests/day at a fintech startup. Led a team of 4 developers. Migrated monolithic Django app to FastAPI microservices, reducing latency by 60%.",
        "bio": "Passionate backend engineer with a focus on scalable systems and clean code. Open source contributor (FastAPI, SQLAlchemy). Looking for senior/staff roles at product companies where I can have architecture-level impact.",
        "education": "BS Computer Science, UC Berkeley",
        "years_experience": 6,
    },
    {
        "id": "cand_2",
        "name": "Priya Sharma",
        "skills": "Python, Flask, Django REST Framework, PostgreSQL, MongoDB, Docker, GCP, Pub/Sub, BigQuery",
        "experience": "4 years backend development. Built data pipelines processing 100GB/day at a healthcare company. Designed RESTful APIs serving 500K daily users. Implemented HIPAA-compliant data storage and encryption.",
        "bio": "Backend developer passionate about clean architecture and healthcare technology. Interested in roles where I can use technology to improve patient outcomes. Seeking mid-to-senior positions.",
        "education": "MS Computer Science, Stanford University",
        "years_experience": 4,
    },
    # Frontend
    {
        "id": "cand_3",
        "name": "Marcus Johnson",
        "skills": "React, TypeScript, Next.js, Tailwind CSS, GraphQL, Node.js, Figma, Storybook, Cypress",
        "experience": "4 years frontend development. Built 3 production apps from scratch using React. Created a shared component library used by 5 teams. Strong design sensibility and component architecture skills. Performance optimization specialist.",
        "bio": "Frontend engineer who loves building beautiful, accessible UIs. I bridge the gap between design and development. Contributed to Material UI. Looking for roles at design-forward companies or as a founding frontend engineer.",
        "education": "BS Computer Science, Georgia Tech",
        "years_experience": 4,
    },
    {
        "id": "cand_4",
        "name": "Sarah Kim",
        "skills": "Vue.js, Nuxt.js, TypeScript, SCSS, D3.js, Webpack, Jest, Playwright, Accessibility (WCAG 2.1)",
        "experience": "5 years frontend development, 2 years as tech lead. Built real-time trading dashboards for a fintech company. Led accessibility initiative achieving WCAG 2.1 AA compliance across the platform. Mentored 3 junior developers.",
        "bio": "Frontend tech lead passionate about web performance and accessibility. Strong advocate for user-centered design. Seeking engineering manager or staff frontend roles at companies building complex, data-rich applications.",
        "education": "BS Information Systems, Carnegie Mellon",
        "years_experience": 5,
    },
    # Data Science / ML
    {
        "id": "cand_5",
        "name": "Dr. James Wright",
        "skills": "Python, TensorFlow, PyTorch, scikit-learn, SQL, Apache Spark, MLflow, Hugging Face, LangChain, RAG",
        "experience": "5 years in data science and ML. Published 2 papers on NLP at ACL/EMNLP. Built recommendation systems serving 1M+ users with 15% uplift in engagement. Experience with A/B testing at scale. Recently focused on LLM applications and RAG systems.",
        "bio": "Data scientist with a PhD in Computer Science (NLP specialization). Interested in NLP, recommendation systems, and applying ML to real-world problems. Seeking Staff ML Engineer or Research Scientist roles.",
        "education": "PhD Computer Science (NLP), MIT",
        "years_experience": 5,
    },
    {
        "id": "cand_6",
        "name": "Emily Zhang",
        "skills": "Python, R, SQL, Tableau, Looker, Statistics, A/B Testing, Excel, Google Analytics, dbt",
        "experience": "2 years as a data analyst at an e-commerce company. Built executive dashboards tracking $50M revenue pipeline. Designed A/B testing framework that increased conversion by 8%. Automated weekly reporting saving 10 hours/week.",
        "bio": "Analytical thinker transitioning from data analyst to data scientist. Currently completing ML specialization on Coursera. Looking for roles that combine business analytics with machine learning.",
        "education": "BS Mathematics, University of Michigan",
        "years_experience": 2,
    },
    # DevOps / Cloud
    {
        "id": "cand_7",
        "name": "Raj Patel",
        "skills": "AWS, Kubernetes, Terraform, Docker, GitHub Actions, Prometheus, Grafana, Linux, Ansible, Python",
        "experience": "5 years DevOps/SRE. Managed infrastructure for a platform with 99.99% uptime serving 5M users. Reduced cloud costs by 40% through reserved instances and auto-scaling optimization. Led migration from EC2 to EKS (Kubernetes).",
        "bio": "DevOps engineer passionate about automation, reliability, and infrastructure as code. AWS Solutions Architect Professional certified. Looking for challenging environments at scale where I can build self-healing infrastructure.",
        "education": "BS Computer Engineering, University of Texas at Austin",
        "years_experience": 5,
    },
    # Product / Management
    {
        "id": "cand_8",
        "name": "Lisa Anderson",
        "skills": "Product Management, Agile, SQL, Data Analysis, User Research, Roadmapping, Jira, Amplitude, Figma",
        "experience": "7 years product management. Launched 5 products from 0 to 1. Managed cross-functional teams of 15+. Led AI/ML product development that grew from $0 to $5M ARR in 18 months. Experienced in enterprise and consumer products.",
        "bio": "Product leader who turns complex problems into simple, lovable products. Deep experience in AI products and enterprise SaaS. Looking for VP Product or Head of Product roles at growth-stage startups.",
        "education": "MBA, Harvard Business School; BS Engineering, Cornell",
        "years_experience": 7,
    },
    # Mobile
    {
        "id": "cand_9",
        "name": "David Park",
        "skills": "Swift, SwiftUI, UIKit, Objective-C, Core Data, Combine, XCTest, Fastlane, Firebase, ARKit",
        "experience": "6 years iOS development. Built apps with 2M+ downloads. Led iOS team of 3 at a health-tech startup. Implemented real-time video calling, offline-first architecture, and HealthKit integration. App Store featured twice.",
        "bio": "Senior iOS developer focused on building polished, performant mobile experiences. Passionate about SwiftUI and modern iOS architecture. Seeking lead iOS or mobile architect roles at companies shipping to millions of users.",
        "education": "BS Computer Science, University of Washington",
        "years_experience": 6,
    },
    # Fresher / Junior
    {
        "id": "cand_10",
        "name": "Aisha Mohammed",
        "skills": "Python, JavaScript, React, HTML/CSS, Git, SQL, Data Structures, Algorithms, AWS basics",
        "experience": "Recent CS graduate. Completed 3 internships: backend (Python/Flask), frontend (React), and data engineering (ETL pipelines). Built a full-stack capstone project — a job board platform using FastAPI + React. Active on LeetCode (500+ problems solved).",
        "bio": "Enthusiastic new grad looking for my first full-time software engineering role. Quick learner who thrives in fast-paced environments. Open to backend, frontend, or full-stack positions. Particularly interested in Python or React roles.",
        "education": "BS Computer Science, University of Illinois Urbana-Champaign (GPA 3.8)",
        "years_experience": 0,
    },
    # Full Stack / Senior
    {
        "id": "cand_11",
        "name": "Carlos Rodriguez",
        "skills": "Node.js, Express, React, TypeScript, PostgreSQL, MongoDB, Docker, AWS (Lambda, DynamoDB), Terraform, GraphQL",
        "experience": "7 years full-stack development. Tech lead at a Series B startup. Architected a multi-tenant SaaS platform serving 10K+ businesses. Built real-time collaboration features (WebSocket). Hired and mentored 5 engineers.",
        "bio": "Full-stack tech lead who loves building products end-to-end. Strong in system design and team leadership. Seeking Staff Engineer or Engineering Manager roles at mission-driven companies. Open to both startups and established tech companies.",
        "education": "MS Computer Science, University of Toronto",
        "years_experience": 7,
    },
    # Security
    {
        "id": "cand_12",
        "name": "Nicole Thompson",
        "skills": "Penetration Testing, Python, Go, AWS Security, Burp Suite, OWASP, Snyk, SOC2, ISO 27001, Incident Response",
        "experience": "5 years application security. Led security team at a fintech company. Conducted 50+ penetration tests. Built automated SAST/DAST pipeline catching 90% of vulnerabilities before production. Led SOC2 Type II certification process.",
        "bio": "Security engineer passionate about building secure systems from the ground up. CISSP and OSCP certified. Interested in AppSec lead or Security Architect roles at companies handling sensitive financial or healthcare data.",
        "education": "BS Cybersecurity, Purdue University; CISSP, OSCP certifications",
        "years_experience": 5,
    },
]


def get_text_for_entity(entity_type: str, entity: dict) -> str:
    """Extract text content from an entity for embedding."""
    if entity_type == "job_post":
        parts = [
            entity.get("title", ""),
            entity.get("description", ""),
            entity.get("requirements", ""),
            entity.get("location", ""),
            entity.get("company", ""),
        ]
        return " ".join(p for p in parts if p).strip()
    elif entity_type == "company":
        parts = [
            entity.get("name", ""),
            entity.get("description", ""),
            entity.get("industry", ""),
        ]
        return " ".join(p for p in parts if p).strip()
    elif entity_type == "candidate":
        parts = [
            entity.get("name", ""),
            entity.get("skills", ""),
            entity.get("experience", ""),
            entity.get("bio", ""),
            entity.get("education", ""),
        ]
        return " ".join(p for p in parts if p).strip()
    return ""


def seed():
    """Seed ChromaDB with sample data using real embeddings (OpenAI or Gemini based on config)."""
    vector_db_url = os.getenv("VECTOR_DB_URL", "http://localhost:8001")
    llm_provider = os.getenv("LLM_PROVIDER", "google").lower().strip()

    print(f"📡 Connecting to ChromaDB at {vector_db_url}")
    print(f"🤖 Using LLM provider: {llm_provider}")

    # Connect to ChromaDB
    parsed = urlparse(vector_db_url)
    client = chromadb.HttpClient(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8001,
    )

    # Verify connection
    try:
        client.heartbeat()
        print("✅ ChromaDB connection successful\n")
    except Exception as e:
        print(f"❌ Cannot connect to ChromaDB: {e}")
        print("   Make sure ChromaDB is running: chroma run --port 8001")
        sys.exit(1)

    # Initialize embeddings based on provider
    if llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key or openai_api_key.startswith("sk-your"):
            print("❌ ERROR: OPENAI_API_KEY not set in .env file")
            sys.exit(1)

        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        print(f"🤖 Using OpenAI embedding model: {embedding_model}")

        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,
        )
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        google_api_key = os.getenv("GOOGLE_API_KEY", "")
        if not google_api_key or google_api_key == "YOUR_GEMINI_API_KEY_HERE":
            print("❌ ERROR: GOOGLE_API_KEY not set in .env file")
            sys.exit(1)

        embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        print(f"🤖 Using Gemini embedding model: {embedding_model}")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key,
        )

    async def generate_and_seed():
        # Seed job posts
        print(f"\n🔄 Seeding {len(SAMPLE_JOB_POSTS)} job posts...")
        job_collection = client.get_or_create_collection(
            name="job_posts", metadata={"hnsw:space": "cosine"}
        )
        job_texts = [get_text_for_entity("job_post", jp) for jp in SAMPLE_JOB_POSTS]
        job_embeddings = await embeddings.aembed_documents(job_texts)

        job_collection.upsert(
            ids=[f"job_post_{jp['id']}" for jp in SAMPLE_JOB_POSTS],
            embeddings=job_embeddings,
            metadatas=[
                {
                    "entity_type": "job_post",
                    "entity_id": jp["id"],
                    "title": jp["title"],
                    "location": jp.get("location", ""),
                    "company": jp.get("company", ""),
                    "salary_range": jp.get("salary_range", ""),
                    "text_snippet": get_text_for_entity("job_post", jp)[:500],
                }
                for jp in SAMPLE_JOB_POSTS
            ],
            documents=[get_text_for_entity("job_post", jp)[:500] for jp in SAMPLE_JOB_POSTS],
        )
        print(f"   ✅ {len(SAMPLE_JOB_POSTS)} job posts seeded")

        # Seed companies
        print(f"🔄 Seeding {len(SAMPLE_COMPANIES)} companies...")
        company_collection = client.get_or_create_collection(
            name="companies", metadata={"hnsw:space": "cosine"}
        )
        company_texts = [get_text_for_entity("company", c) for c in SAMPLE_COMPANIES]
        company_embeddings = await embeddings.aembed_documents(company_texts)

        company_collection.upsert(
            ids=[f"company_{c['id']}" for c in SAMPLE_COMPANIES],
            embeddings=company_embeddings,
            metadatas=[
                {
                    "entity_type": "company",
                    "entity_id": c["id"],
                    "name": c["name"],
                    "industry": c["industry"],
                    "size": c.get("size", ""),
                    "funding": c.get("funding", ""),
                    "text_snippet": get_text_for_entity("company", c)[:500],
                }
                for c in SAMPLE_COMPANIES
            ],
            documents=[get_text_for_entity("company", c)[:500] for c in SAMPLE_COMPANIES],
        )
        print(f"   ✅ {len(SAMPLE_COMPANIES)} companies seeded")

        # Seed candidates
        print(f"🔄 Seeding {len(SAMPLE_CANDIDATES)} candidates...")
        candidate_collection = client.get_or_create_collection(
            name="candidates", metadata={"hnsw:space": "cosine"}
        )
        candidate_texts = [get_text_for_entity("candidate", c) for c in SAMPLE_CANDIDATES]
        candidate_embeddings = await embeddings.aembed_documents(candidate_texts)

        candidate_collection.upsert(
            ids=[f"candidate_{c['id']}" for c in SAMPLE_CANDIDATES],
            embeddings=candidate_embeddings,
            metadatas=[
                {
                    "entity_type": "candidate",
                    "entity_id": c["id"],
                    "name": c.get("name", ""),
                    "skills": c["skills"],
                    "years_experience": str(c.get("years_experience", "")),
                    "text_snippet": get_text_for_entity("candidate", c)[:500],
                }
                for c in SAMPLE_CANDIDATES
            ],
            documents=[get_text_for_entity("candidate", c)[:500] for c in SAMPLE_CANDIDATES],
        )
        print(f"   ✅ {len(SAMPLE_CANDIDATES)} candidates seeded")

    asyncio.run(generate_and_seed())

    # Summary
    total = len(SAMPLE_JOB_POSTS) + len(SAMPLE_COMPANIES) + len(SAMPLE_CANDIDATES)
    print("\n" + "=" * 60)
    print("🎉 SEED COMPLETE!")
    print(f"   • {len(SAMPLE_JOB_POSTS)} job posts (various roles & levels)")
    print(f"   • {len(SAMPLE_COMPANIES)} companies (diverse industries)")
    print(f"   • {len(SAMPLE_CANDIDATES)} candidates (junior to senior)")
    print(f"   • Total: {total} entities with REAL embeddings ({llm_provider})")
    print("=" * 60)
    print("\n✅ You can now start the backend and test the AI features!")
    print("\nExample queries to try:")
    print('  • GET /ask-ai?query=List top remote Python jobs for a fresher')
    print('  • GET /ask-ai?query=Which companies are hiring for data science?')
    print('  • GET /ask-ai?query=Find senior backend roles paying over $150k')
    print('  • POST /recommend with a Python developer resume')
    print('  • POST /agent/task with "Find cloud computing jobs and summarize them"')


if __name__ == "__main__":
    seed()
