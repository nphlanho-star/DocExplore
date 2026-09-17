# DocExplore - Project Planning

## 1. Project Overview

DocExplore is a web-based document question-answering system using **Retrieval-Augmented Generation (RAG)**.

The main goal is to allow users to upload, manage, search, and ask questions about a large collection of documents with different formats and contents.

The system retrieves relevant information from the document collection before generating an answer with an LLM. The answer should be grounded in the provided documents and include references to the original sources whenever possible.

## 2. Problem

Working with a large document collection creates several challenges:

* Documents can have different formats such as PDF, DOCX, XLSX, and PPTX.
* Documents can be very long.
* Some documents are scanned images and require OCR.
* Important information may be stored in tables or complex layouts.
* Naive text chunking can separate related information.
* Searching only by semantic similarity may miss exact keywords.
* Retrieved documents may contain irrelevant information.
* Documents can be updated, duplicated, or conflicting.
* The LLM may generate information that does not exist in the source documents.
* Users need to know where an answer came from.

Therefore, the project focuses not only on building a chatbot, but also on designing a reliable document processing and retrieval pipeline.

## 3. Planning Documents

### Features, Pain Points and Edge Cases

`features-pain-points-edge-cases.md`

Defines the main system features, identifies important pain points, and lists edge cases that the system needs to handle.

### Solution Directions

`solution-directions.md`

Proposes technical directions for solving the identified problems, including document processing, chunking, retrieval, reranking, grounding, citation, and document management.

### System Workflow

`workflow.mmd`

Contains the proposed end-to-end workflow of the DocExplore RAG system using Mermaid.

## 4. Main System Flow

```text
User
  ↓
Upload Documents
  ↓
Document Processing
  ↓
Text Extraction / OCR
  ↓
Chunking
  ↓
Embedding & Indexing
  ↓
User Query
  ↓
Retrieval
  ↓
Reranking
  ↓
Context Assembly
  ↓
LLM Generation
  ↓
Answer + Source Citation
```

## 5. Planning Goal

The purpose of this planning phase is to define a practical architecture for a scalable and reliable RAG system before implementation begins.
