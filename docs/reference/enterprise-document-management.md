# Enterprise Document Management — source material

Prepared with Gemini by the project owner (23 Nov 2026). Kept as the
product-layer reference for the document system: physical storage,
logical organisation, discovery, security and deployment.

It is **source material, not a decision of this library**. The library-side
design is in [document store](document-store.md); this document describes the
system the local adapter is meant to be able to grow into.

---

# Comprehensive System Requirements Specification: Enterprise AI-Native Document Management

## 1. Executive Summary & Core Problem Statement
Traditional business document organization relies on static, deterministic folder hierarchies mapped to organizational charts (e.g., `Department > Team > Project > File`). At enterprise scale, this model breaks down due to:
* **Discovery Blind Spots:** Users and AI agents frequently do not know which department or person "owns" a given document, making path-based navigation impossible.
* **Taxonomy & Hierarchy Rot:** Over time, folder structures become bloated, ambiguous, and incomprehensible.
* **Permission Debt & Org Chart Shifts:** When people change teams or projects, static folder permissions fail to update, creating security leaks or access bottlenecks.
* **Cross-Functional Overlap:** Documents rarely belong to a single silo; they may be relevant to multiple teams simultaneously, or require strict isolation (e.g., HR-only documents) while still existing within the broader corporate knowledge base.
* **Vendor & Format Lock-in:** Businesses cannot risk trapping their institutional knowledge in proprietary databases or formats where files cannot be easily reclaimed.

**The Goal:** Design an open-source, locally runnable, programmatically accessible document organization and discovery system capable of managing an entire organization's work product ($10^5$ to $10^7+$ files) for both human users and autonomous AI agents.

---

## 2. Core Architectural Principles

### A. Decoupled Multi-Layer Architecture
To avoid hierarchy rot and vendor lock-in, physical storage must be completely decoupled from logical organization and semantic discovery:
1. **The Physical Archive (Object Storage):** Files are stored in a flat, S3-compatible object store (e.g., MinIO). There are no deep OS-level nested directories, preventing file-count bottlenecks and ensuring raw files remain easily reclaimable in open formats.
2. **The Logical Controller (Metadata & Versioning Layer):** An API-first Document Management System (e.g., FormKiQ Core) acts as the system of record. It assigns Universal Unique Identifiers (UUIDs), enforces metadata schemas, manages version history, and handles "Release" snapshots.
3. **The Discovery Brain (Permission-Aware RAG / Vector Index):** A single, multi-tenant semantic search engine (e.g., Onyx) indexes document content and metadata stamps to support vague, natural-language discovery while strictly enforcing access controls.

### B. Poly-Hierarchy & Virtual Organization ("Label, Don't Folder")
* A document does not live in a single static location. Instead, it is defined by rich, multi-dimensional metadata attributes (e.g., `project`, `department`, `status`, `sensitivity`, `document_type`).
* Logical "folders" or views are dynamically generated on-the-fly based on metadata queries. If an attribute changes (e.g., `status: draft` $\rightarrow$ `status: final`), the document's logical grouping updates automatically without moving the underlying file.

---

## 3. Content, Linking & Versioning Requirements

### A. Heterogeneous File Support & Atomic Markdown
* **Atomic Markdown Fragments:** A substantial portion of the corpus will consist of small, modular Markdown (`.md`) files—for example, a business plan broken down so each section (Executive Summary, Financials, Marketing) is an independent file.
* **Legacy & Binary Office Files:** The system must equally ingest, parse, OCR, and index unstructured binary files, including PDFs, Word documents (`.docx`), Spreadsheets (`.xlsx`, `.csv`), Presentations (`.pptx`), and images.
* **Unified Treatment:** Both atomic Markdown snippets and binary office files must be treated as first-class citizens with UUIDs, metadata, and version histories.

### B. Programmatic Linking & Graph Composition
* **UUID-Based Identity:** Every document fragment receives a permanent, globally unique UUID upon creation.
* **Manifest & Graph Linking:** Agents and users link work products using UUIDs rather than fragile file paths (e.g., embedding `urn:uuid:{UUID}` in Markdown or storing `references: [UUID_1, UUID_2]` in metadata).
* **Virtual Assembly:** Complex multi-part documents (like a business plan) are assembled via a lightweight "Manifest" document or metadata query that points to the constituent fragment UUIDs.

### C. Versioning & Point-in-Time "Release" Tagging
* **Immutable Releases:** Users and agents must be able to "tag a release" of a multi-part document or collection (e.g., `Business-Plan-v1.1`).
* **Snapshot Mechanics:** A release tag does not duplicate physical files. It creates an immutable metadata snapshot mapping the release name to the exact `UUID + VersionID` pairs of all constituent documents (both Markdown fragments and binary files) as they existed at that moment.
* **Auditability & Evolution:** Working drafts can continue to evolve independently after a release is tagged, while querying `release: v1.1` deterministically returns the exact frozen state.

---

## 4. Retrieval & Discovery Requirements

The system must support two distinct modes of retrieval:

### A. Deterministic Programmatic Retrieval (For Exact Agent Workflows)
* Agents must be able to query the system via REST API/CLI using exact metadata filters, UUIDs, and release tags (e.g., `GET /documents?project=alpha&section=finance&release=v1.0`).
* Must return exact, deterministic matches with zero probabilistic guesswork.

### B. Hybrid Semantic Discovery (For Vague Human/Agent Intent)
* When a user or agent has vague knowledge of what exists (e.g., *"Do we have brand guidelines?"* or *"What is our policy on customer refunds?"*), keyword and folder searches fail.
* The system must use **Hybrid Search** (combining vector embeddings for semantic meaning with keyword matching and metadata boosting) to surface relevant fragments even if the user's terminology doesn't match the document titles.
* An LLM synthesis layer acts as a "librarian" to summarize findings across disparate fragments (e.g., combining a Markdown voice guide, a PDF logo spec, and an Excel color palette) and return direct UUID citations.

---

## 5. Security, Privacy & Multi-Tenancy Requirements

### A. Single-Instance Multi-Tenancy
* The organization must not require separate RAG/DMS instances per user or team. A single unified index and storage layer must serve the entire company.

### B. Document-Level Security (DLS) & Early-Filtering
* **Zero-Leak Guarantee:** Sensitive data (e.g., HR payroll, executive restructuring) must be strictly isolated from unauthorized users, even during broad semantic searches or AI summarizations.
* **Early-Filtering at the Index Level:** Every vector chunk and metadata record must be stamped with Access Control Lists (`allow_groups`, `allow_users`). When a user or agent queries the system, their identity context (via OIDC/Keycloak) is injected directly into the database query *before* semantic similarity is calculated or any text is passed to an LLM.
* **Link & Snapshot Protection:** If a public document or release manifest contains a UUID link to an HR-private fragment, resolving that UUID as a non-HR user must fail (`403 Forbidden` / `Not Found`).

### C. Attribute-Based Access Control (ABAC)
* Access rights should dynamically evaluate user attributes (group memberships, roles, project assignments) against document metadata, ensuring that when the org chart changes, permissions update automatically via the Identity Provider without manual file-level permission edits.

---

## 6. Programmatic Access, Scale & Deployment Requirements

### A. 100% API-First Design
* Every operation—uploading files, assigning/updating metadata, linking UUIDs, tagging releases, updating ACLs, and querying—must be executable programmatically via REST API or CLI for autonomous AI agents.

### B. Enterprise Scale Performance
* Must handle organizational scale ($10^5$ to $10^7$ documents, mixing millions of small Markdown files with large binaries) without search latency or storage bottlenecks.

### C. Open-Source, Local Execution & CI/CD Automation
* **No Proprietary Lock-in:** All components must be open-source, storing raw files in open formats so data can be reclaimed or migrated at any time.
* **Local-First & IaC:** Must be runnable locally on a developer machine and spin up automatically within a **GitHub Actions** CI/CD pipeline (under 5 minutes on a standard Linux runner) for automated integration testing of agent workflows.
* **Declarative Configuration:** Infrastructure (buckets, IAM policies, user groups, search connectors) must be scriptable via **Docker Compose**, **Terraform**, and admin CLIs.

---

## 7. Reference Implementation Stack (For Evaluation & Testing)
Any agent or architect evaluating or building this system should benchmark against this reference open-source stack:
1. **Storage Layer:** **MinIO** (S3-compatible object storage for raw `.md`, `.pdf`, `.docx`, `.xlsx` files).
2. **Logic & Versioning Layer:** **FormKiQ Core** (API-first DMS handling UUIDs, metadata schemas, versioning, and release collection snapshots).
3. **Discovery & Security Layer:** **Onyx (formerly Danswer)** (Multi-format parsing via `Unstructured`, hybrid vector search, LLM synthesis, and Document-Level Security).
4. **Identity & Access Layer:** **Keycloak** (OIDC identity provider managing user groups like `General` vs. `HR` for dynamic ABAC filtering).
5. **Orchestration:** **Docker Compose** + **Terraform** (For local execution and GitHub Actions CI/CD testing).