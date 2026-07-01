"""Sync API endpoint for the AI Intelligence Layer.

Provides the POST /sync/full endpoint that triggers a full idempotent
re-sync of all entity embeddings to the vector database.

This version uses an in-memory entity fetcher with sample data,
bypassing PostgreSQL for demonstration purposes.
"""

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.ai.config import AIConfig
from app.ai.embedding_service import EmbeddingService
from app.ai.models import SyncReport, VectorDocument
from app.ai.vectorstore.factory import create_vector_store
from app.ai.vectorstore import VectorStoreInterface

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])


def _get_config() -> AIConfig:
    """Provide AIConfig instance."""
    return AIConfig()


def _get_entity_text(entity_type: str, entity: Dict[str, Any]) -> str:
    """Extract and combine relevant text fields from an entity for embedding."""
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


def _get_sample_data() -> Dict[str, List[Dict[str, Any]]]:
    """Return sample data for syncing embeddings.

    Inline data avoids import path issues across different deployment environments.
    """
    return {
        "job_post": [
            {"id": "job_1", "title": "Senior Python Developer", "description": "We are looking for a Senior Python Developer to join our backend team. You will design and build scalable microservices, mentor junior developers, and work closely with the DevOps team to deploy services on AWS.", "requirements": "5+ years Python experience, FastAPI or Django, PostgreSQL, Docker, AWS, CI/CD pipelines.", "location": "Remote", "company": "TechNova Solutions"},
            {"id": "job_2", "title": "Junior Python Backend Developer", "description": "Join our growing engineering team as a Junior Python Developer. You will work on REST API development, write unit tests, and learn from senior engineers.", "requirements": "1+ years Python, basic knowledge of Flask or FastAPI, SQL fundamentals, Git.", "location": "Remote", "company": "DataStream Analytics"},
            {"id": "job_3", "title": "Python Django Developer", "description": "We need a mid-level Django developer to maintain and extend our e-commerce platform. Build new features, optimize database queries, integrate payment gateways.", "requirements": "3+ years Django, PostgreSQL, REST APIs, Celery, Redis.", "location": "New York, NY (Hybrid)", "company": "ShopFlow Inc."},
            {"id": "job_4", "title": "Frontend React Engineer", "description": "Join our product team to build beautiful, responsive web applications using React and TypeScript. Collaborate with designers and backend engineers.", "requirements": "3+ years React/TypeScript, Next.js, Tailwind CSS, REST APIs, Git.", "location": "San Francisco, CA", "company": "PixelCraft Design Studio"},
            {"id": "job_5", "title": "Senior Frontend Developer (Vue.js)", "description": "Lead frontend development for our fintech dashboard. Build complex data visualizations, real-time trading interfaces.", "requirements": "5+ years frontend, 3+ years Vue.js/Nuxt.js, TypeScript, D3.js.", "location": "London, UK (Remote OK)", "company": "DataStream Analytics"},
            {"id": "job_6", "title": "Data Scientist - Machine Learning", "description": "Develop and deploy ML models for our recommendation engine. Work with large datasets, build predictive models using deep learning.", "requirements": "MS/PhD in CS or Statistics, Python, scikit-learn, TensorFlow/PyTorch, SQL.", "location": "Remote", "company": "TechNova Solutions"},
            {"id": "job_7", "title": "Junior Data Analyst", "description": "Analyze business metrics, create dashboards, write SQL queries to extract insights. Perfect for recent graduates who love working with data.", "requirements": "Bachelor's in Statistics or CS. SQL proficiency, Excel, basic Python or R.", "location": "Austin, TX", "company": "GreenLeaf HealthTech"},
            {"id": "job_8", "title": "NLP Engineer", "description": "Build and deploy NLP models for customer support automation. Work on intent classification, named entity recognition, sentiment analysis.", "requirements": "3+ years NLP/ML, Python, Hugging Face Transformers, PyTorch.", "location": "Remote", "company": "TechNova Solutions"},
            {"id": "job_9", "title": "DevOps Engineer", "description": "Manage cloud infrastructure on AWS. Automate deployments using Terraform, manage Kubernetes clusters, set up monitoring.", "requirements": "3+ years DevOps, AWS, Kubernetes, Terraform, Docker, CI/CD.", "location": "Berlin, Germany", "company": "CloudPeak Infrastructure"},
            {"id": "job_10", "title": "Cloud Solutions Architect", "description": "Design and implement cloud-native architectures for enterprise clients. Lead technical discussions, create architecture diagrams.", "requirements": "7+ years software engineering, 3+ years cloud architecture, AWS/Azure/GCP certifications.", "location": "Chicago, IL (Hybrid)", "company": "CloudPeak Infrastructure"},
            {"id": "job_11", "title": "Site Reliability Engineer (SRE)", "description": "Keep our platform running at scale (10M daily active users). Design fault-tolerant systems, automate incident response.", "requirements": "4+ years SRE or DevOps, Linux, Python/Go, Kubernetes, observability tools.", "location": "Remote (US timezone)", "company": "TechNova Solutions"},
            {"id": "job_12", "title": "Full Stack Developer (Node.js + React)", "description": "Build end-to-end features for our SaaS platform serving 50K+ businesses. Work across the full stack.", "requirements": "3+ years full stack, Node.js, Express, React, TypeScript, PostgreSQL.", "location": "Toronto, Canada (Hybrid)", "company": "ShopFlow Inc."},
            {"id": "job_13", "title": "Full Stack Python Developer", "description": "Work on our healthcare platform from frontend to backend. Build patient-facing React dashboards, develop FastAPI microservices.", "requirements": "3+ years full stack, Python (FastAPI/Django), React/TypeScript, PostgreSQL.", "location": "Boston, MA", "company": "GreenLeaf HealthTech"},
            {"id": "job_14", "title": "Mobile Developer - iOS (Swift)", "description": "Develop and maintain our iOS application used by 500K+ users. Implement new features using SwiftUI.", "requirements": "4+ years iOS development, Swift, UIKit/SwiftUI, Core Data.", "location": "Seattle, WA", "company": "PixelCraft Design Studio"},
            {"id": "job_15", "title": "Android Developer (Kotlin)", "description": "Build and maintain our Android app for telemedicine platform. Work with Jetpack Compose, implement real-time video calling.", "requirements": "3+ years Android development, Kotlin, Jetpack Compose, MVVM.", "location": "Remote", "company": "GreenLeaf HealthTech"},
            {"id": "job_16", "title": "Product Manager - AI Products", "description": "Lead the product strategy for AI-powered recruitment features. Define roadmaps, prioritize features based on data.", "requirements": "5+ years product management, experience shipping AI/ML products.", "location": "San Francisco, CA", "company": "TechNova Solutions"},
            {"id": "job_17", "title": "Engineering Manager - Backend", "description": "Manage a team of 8 backend engineers. Conduct 1:1s, drive technical strategy, hire and mentor engineers.", "requirements": "5+ years engineering, 2+ years people management.", "location": "New York, NY", "company": "TechNova Solutions"},
            {"id": "job_18", "title": "QA Automation Engineer", "description": "Build and maintain automated testing framework covering web, API, and mobile platforms.", "requirements": "3+ years QA automation, Cypress or Playwright, Python or JavaScript.", "location": "Remote", "company": "ShopFlow Inc."},
            {"id": "job_19", "title": "Senior UI/UX Designer", "description": "Design intuitive interfaces for our B2B analytics platform. Conduct user research, create wireframes and prototypes.", "requirements": "5+ years UI/UX design, Figma expert, design systems experience.", "location": "Remote", "company": "DataStream Analytics"},
            {"id": "job_20", "title": "Application Security Engineer", "description": "Protect our platform and customer data. Conduct code reviews for security vulnerabilities, perform penetration testing.", "requirements": "4+ years application security, OWASP Top 10, Python/Go.", "location": "Remote (US)", "company": "TechNova Solutions"},
            {"id": "job_21", "title": "Senior Data Engineer", "description": "Build and maintain our data platform processing 500GB+ daily. Design ETL pipelines using Apache Airflow.", "requirements": "5+ years data engineering, Python, SQL, Apache Airflow, Spark, Snowflake.", "location": "Remote", "company": "DataStream Analytics"},
            {"id": "job_22", "title": "Blockchain Developer (Solidity)", "description": "Develop smart contracts for our DeFi platform. Write and audit Solidity code, build integration tests.", "requirements": "2+ years Solidity, Ethereum/EVM, Hardhat/Foundry, OpenZeppelin.", "location": "Remote (Global)", "company": "CryptoVault Labs"},
            {"id": "job_23", "title": "AI/ML Engineer - LLM Applications", "description": "Build production LLM-powered features. Implement RAG pipelines, fine-tune models, build evaluation frameworks.", "requirements": "3+ years ML engineering, Python, LangChain, OpenAI API, vector databases.", "location": "Remote", "company": "TechNova Solutions"},
            {"id": "job_24", "title": "Software Engineering Intern (Summer 2025)", "description": "12-week paid internship. Work on real features alongside senior engineers, attend tech talks, receive mentorship.", "requirements": "Currently pursuing BS/MS in Computer Science. Knowledge of Python, Java, or JavaScript.", "location": "San Francisco, CA (On-site)", "company": "TechNova Solutions"},
            {"id": "job_25", "title": "Cloud Security Analyst", "description": "Monitor and secure AWS cloud infrastructure. Configure GuardDuty, Security Hub, perform threat modeling.", "requirements": "3+ years cloud security, AWS, scripting (Python/Bash), CIS benchmarks.", "location": "Washington, DC (Hybrid)", "company": "CloudPeak Infrastructure"},
        ],
        "company": [
            {"id": "comp_1", "name": "TechNova Solutions", "description": "Fast-growing SaaS company building AI-powered tools for HR and recruitment. 200+ employees, serves Fortune 500 clients.", "industry": "Enterprise SaaS / HR Tech"},
            {"id": "comp_2", "name": "DataStream Analytics", "description": "Specializes in big data processing and real-time analytics platforms for e-commerce and fintech companies.", "industry": "Data Analytics / FinTech"},
            {"id": "comp_3", "name": "CloudPeak Infrastructure", "description": "Provides managed cloud services and DevOps consulting. Helped 300+ companies migrate to the cloud.", "industry": "Cloud Computing / Consulting"},
            {"id": "comp_4", "name": "PixelCraft Design Studio", "description": "Digital design and development agency focused on creating beautiful mobile and web experiences.", "industry": "Design Agency / Mobile Development"},
            {"id": "comp_5", "name": "GreenLeaf HealthTech", "description": "Builds telemedicine and health monitoring platforms used by 200+ hospitals and clinics across the US.", "industry": "HealthTech / Telemedicine"},
            {"id": "comp_6", "name": "ShopFlow Inc.", "description": "Headless commerce platform powering 50,000+ online stores worldwide. Processes $2B+ in GMV annually.", "industry": "E-Commerce / SaaS"},
            {"id": "comp_7", "name": "CryptoVault Labs", "description": "Builds secure DeFi infrastructure for institutional investors. $500M+ in total value locked.", "industry": "Blockchain / DeFi"},
            {"id": "comp_8", "name": "EduSpark Learning", "description": "EdTech company building adaptive learning platform using AI. Partners with 50+ universities, 2M+ active learners.", "industry": "EdTech / AI Learning"},
        ],
        "candidate": [
            {"id": "cand_1", "name": "Alex Chen", "skills": "Python, FastAPI, Django, PostgreSQL, Docker, AWS, Redis, Celery, Kubernetes", "experience": "6 years backend developer. Built microservices handling 10M+ requests/day.", "bio": "Passionate backend engineer focused on scalable systems.", "education": "BS Computer Science, UC Berkeley"},
            {"id": "cand_2", "name": "Priya Sharma", "skills": "Python, Flask, Django REST Framework, PostgreSQL, MongoDB, Docker, GCP", "experience": "4 years backend development. Built data pipelines processing 100GB/day.", "bio": "Backend developer passionate about healthcare technology.", "education": "MS Computer Science, Stanford University"},
            {"id": "cand_3", "name": "Marcus Johnson", "skills": "React, TypeScript, Next.js, Tailwind CSS, GraphQL, Node.js, Figma", "experience": "4 years frontend development. Built 3 production apps from scratch.", "bio": "Frontend engineer who loves building beautiful, accessible UIs.", "education": "BS Computer Science, Georgia Tech"},
            {"id": "cand_4", "name": "Sarah Kim", "skills": "Vue.js, Nuxt.js, TypeScript, SCSS, D3.js, Playwright, Accessibility", "experience": "5 years frontend, 2 years as tech lead. Built real-time trading dashboards.", "bio": "Frontend tech lead passionate about web performance and accessibility.", "education": "BS Information Systems, Carnegie Mellon"},
            {"id": "cand_5", "name": "Dr. James Wright", "skills": "Python, TensorFlow, PyTorch, scikit-learn, SQL, Spark, MLflow, LangChain, RAG", "experience": "5 years data science and ML. Published 2 papers at ACL/EMNLP.", "bio": "Data scientist with PhD in NLP. Seeking Staff ML Engineer roles.", "education": "PhD Computer Science (NLP), MIT"},
            {"id": "cand_6", "name": "Emily Zhang", "skills": "Python, R, SQL, Tableau, Looker, Statistics, A/B Testing, dbt", "experience": "2 years data analyst. Built executive dashboards tracking $50M revenue.", "bio": "Transitioning from data analyst to data scientist.", "education": "BS Mathematics, University of Michigan"},
            {"id": "cand_7", "name": "Raj Patel", "skills": "AWS, Kubernetes, Terraform, Docker, GitHub Actions, Prometheus, Grafana, Linux", "experience": "5 years DevOps/SRE. Managed infrastructure with 99.99% uptime.", "bio": "DevOps engineer passionate about automation and reliability.", "education": "BS Computer Engineering, UT Austin"},
            {"id": "cand_8", "name": "Lisa Anderson", "skills": "Product Management, Agile, SQL, Data Analysis, User Research, Roadmapping, Jira", "experience": "7 years product management. Launched 5 products from 0 to 1.", "bio": "Product leader with deep experience in AI products and enterprise SaaS.", "education": "MBA, Harvard Business School"},
            {"id": "cand_9", "name": "David Park", "skills": "Swift, SwiftUI, UIKit, Core Data, Combine, XCTest, Fastlane, Firebase", "experience": "6 years iOS development. Built apps with 2M+ downloads.", "bio": "Senior iOS developer focused on polished mobile experiences.", "education": "BS Computer Science, University of Washington"},
            {"id": "cand_10", "name": "Aisha Mohammed", "skills": "Python, JavaScript, React, HTML/CSS, Git, SQL, Data Structures, Algorithms", "experience": "Recent CS graduate. Completed 3 internships. Built full-stack capstone project.", "bio": "Enthusiastic new grad looking for first full-time role.", "education": "BS Computer Science, UIUC (GPA 3.8)"},
            {"id": "cand_11", "name": "Carlos Rodriguez", "skills": "Node.js, Express, React, TypeScript, PostgreSQL, MongoDB, Docker, AWS, GraphQL", "experience": "7 years full-stack. Tech lead at Series B startup. Architected multi-tenant SaaS.", "bio": "Full-stack tech lead who loves building products end-to-end.", "education": "MS Computer Science, University of Toronto"},
            {"id": "cand_12", "name": "Nicole Thompson", "skills": "Penetration Testing, Python, Go, AWS Security, Burp Suite, OWASP, SOC2", "experience": "5 years application security. Led security team at fintech company.", "bio": "Security engineer passionate about building secure systems.", "education": "BS Cybersecurity, Purdue University; CISSP, OSCP"},
        ],
    }


# Entity type to collection name mapping
_ENTITY_COLLECTION_MAP = {
    "job_post": "job_posts",
    "company": "companies",
    "candidate": "candidates",
}


@router.post(
    "/full",
    response_model=SyncReport,
    status_code=status.HTTP_200_OK,
    summary="Trigger full embedding re-sync",
    description=(
        "Triggers an idempotent full re-sync of all entity embeddings to "
        "the vector database. Creates embeddings for all entities and stores "
        "them in ChromaDB. This operation is idempotent."
    ),
)
async def full_sync(
    config: AIConfig = Depends(_get_config),
) -> SyncReport:
    """Trigger a full idempotent re-sync of all embeddings.

    This simplified version directly embeds and upserts all sample data
    without requiring PostgreSQL. Suitable for demo/deployment.
    """
    start_time = time.time()

    embedding_service = EmbeddingService(config=config)
    vector_store = create_vector_store(config=config)

    all_entities = _get_sample_data()
    created = 0
    failed = 0
    total = 0

    for entity_type, entities in all_entities.items():
        collection = _ENTITY_COLLECTION_MAP.get(entity_type)
        if not collection:
            continue

        for entity in entities:
            total += 1
            entity_id = str(entity.get("id", ""))
            if not entity_id:
                failed += 1
                continue

            try:
                text_content = _get_entity_text(entity_type, entity)
                if not text_content:
                    failed += 1
                    continue

                # Generate embedding
                embedding = await embedding_service.generate_embedding(text_content)

                # Build metadata
                metadata = {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "text_snippet": text_content[:500],
                }

                # Add extra metadata based on type
                if entity_type == "job_post":
                    metadata["title"] = entity.get("title", "")
                    metadata["location"] = entity.get("location", "")
                    metadata["company"] = entity.get("company", "")
                elif entity_type == "company":
                    metadata["name"] = entity.get("name", "")
                    metadata["industry"] = entity.get("industry", "")
                elif entity_type == "candidate":
                    metadata["name"] = entity.get("name", "")
                    metadata["skills"] = entity.get("skills", "")

                # Upsert to vector DB
                doc = VectorDocument(
                    id=f"{entity_type}_{entity_id}",
                    embedding=embedding,
                    metadata=metadata,
                    text_snippet=text_content[:500],
                )
                await vector_store.upsert(collection, [doc])
                created += 1

            except Exception as exc:
                logger.error("Failed to sync %s/%s: %s", entity_type, entity_id, str(exc))
                failed += 1

    duration = time.time() - start_time

    return SyncReport(
        total_entities=total,
        created=created,
        updated=0,
        deleted=0,
        failed=failed,
        duration_seconds=round(duration, 3),
    )
