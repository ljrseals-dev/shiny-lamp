# Repository Placement Correction Record

**Status:** Documentation-only correction  
**Date:** 2026-11-19  
**Repository:** `ljrseals-dev/shiny-lamp`

## Purpose

This record documents a repository-placement error so that future contributors can understand what was misplaced, why it was redirected, and where the authoritative work now lives.

The earlier model-feasibility and benchmark-lifecycle research was placed in this repository under PR #1:

- [Earlier PR #1 — research/llm-benchmark-math-evidence](https://github.com/ljrseals-dev/shiny-lamp/pull/1)

That placement did not match the responsibility of this repository. The material concerns model discovery, artifact acquisition, integrity and provenance verification, memory feasibility, and benchmark eligibility. Those are model-registry and model-distribution lifecycle responsibilities, not the core purpose of this repository.

## Correct destination

The authoritative documentation proposal is now maintained in:

- [ljrseals-dev/LLM-models PR #3 — Document model feasibility and benchmark eligibility lifecycle](https://github.com/ljrseals-dev/LLM-models/pull/3)

The proposed document is:

- `docs/CONFIGURATION-FEASIBILITY-LIFECYCLE.md`

LLM-models is the correct owner because it manages model catalog/discovery, artifact acquisition, integrity and provenance, quarantine, derived-artifact lineage, and model lifecycle state.

## Responsibility boundaries

| Repository | Responsibility |
|---|---|
| `LLM-models` | Authoritative model catalog, artifact lifecycle, integrity/provenance, feasibility and benchmark-eligibility documentation. |
| `LLM-zip-pdf` | Downstream document-ingestion workload and its own extraction, retrieval, citation, and coverage tests. |
| `StackAlign` | Optional host infrastructure and security integration; it does not own the model registry or automatically download models. |
| `shiny-lamp` | No longer the authoritative home for this model-feasibility lifecycle material. |

## Correction and scope

- The earlier PR in this repository was closed without merging.
- The correction does not copy the model-lifecycle specification into this repository or create a competing source of truth.
- The authoritative proposal in LLM-models PR #3 remains open and unmerged unless its GitHub status changes.
- No model was downloaded, installed, executed, or benchmarked as part of this documentation correction.
- No hardware compatibility or performance result is asserted.
- No changes to this repository's `main` branch are made by this PR.

## Naming and affiliation clarification

The repository has been referred to by its maintainer as the “FBI” repository. This document records repository organization only. It does not claim that this GitHub repository is owned, operated, reviewed, endorsed, or monitored by the Federal Bureau of Investigation or any other government agency. This PR does not send a report or notification to the FBI.

## Acceptance criteria

- [x] Record the earlier misplacement and closed PR.
- [x] Identify LLM-models PR #3 as the authoritative destination.
- [x] State downstream and infrastructure ownership boundaries.
- [x] Avoid duplicating the authoritative lifecycle specification.
- [x] Make no claims of external agency notification, model execution, or benchmark results.
