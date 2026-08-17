---
name: product-requirements
description: Elicit and produce testable product feature requirements with functional criteria, modern quality attributes, acceptance criteria, and guidance for choosing User Story, Job Story, Use Case, or atomic requirement formats.
---

# Product Requirements

Use this skill to create, refine, or review requirements for a product feature.

The goal is a **decision-ready and verifiable requirements document**, not a long specification by default. Scale the detail to the feature's risk, complexity, novelty, and number of teams involved.

## 1. Discovery before specification

Establish the following before finalizing requirements:

1. **Problem**: What user or business problem exists, and what evidence supports it?
2. **Outcome**: What observable change should the feature create?
3. **Users and stakeholders**: Who uses, buys, operates, supports, governs, or is affected by it?
4. **Context**: In which situations, channels, environments, and workflows does the need occur?
5. **Scope**: What is included, excluded, deferred, or intentionally unchanged?
6. **Success**: Which leading, lagging, guardrail, and operational measures determine success?
7. **Constraints**: Which legal, policy, platform, budget, schedule, compatibility, or architectural constraints are fixed?
8. **Risks and unknowns**: Which assumptions require validation?

Ask questions in small, decision-oriented groups. Do not ask for information already available in the repository or supplied context.

## 2. Choose the right requirement expression

User Stories are useful, but they are not a universal requirements format. Recommend a format based on the nature of the behavior.

| Situation | Preferred format | Template |
|---|---|---|
| User-visible capability with a clear beneficiary and value | User Story | `As a <role>, I want <capability>, so that <outcome>.` |
| Motivation depends more on a situation than a stable persona | Job Story | `When <situation>, I want to <motivation/action>, so I can <expected outcome>.` |
| Multi-step interaction, alternate flows, permissions, or complex failures | Use Case / scenario | Actor, preconditions, trigger, main flow, alternate flows, postconditions |
| Business rule, interface contract, data rule, compliance rule, or system behavior | Atomic requirement | `REQ-ID — The <system/component> MUST <observable behavior> [under <condition>].` |
| Quality attribute or operating target | Quality scenario | Source, stimulus, environment, artifact, response, measurable response |
| Concrete acceptance example | Gherkin | `Given <context>, When <event>, Then <observable result>.` |

### User Story decision rule

Use a User Story only when all are true:

- a meaningful user or stakeholder role can be named;
- the capability provides identifiable value to that role;
- the story can be completed as a coherent vertical slice;
- acceptance criteria can describe observable behavior without embedding implementation.

Otherwise, use the more suitable format above.

### Recommended User Story format

Default to:

```text
US-<number> — <short title>
As a <specific role>,
I want <capability>,
so that <measurable or observable outcome>.

Acceptance criteria:
- AC-<number>: Given ..., when ..., then ...
```

Rules:

- The role describes a relevant need or permission, not a generic "user".
- The middle clause states the capability, not a screen or technical solution.
- The benefit states the intended outcome and must not merely repeat the capability.
- Check that delivery stories are Independent, Negotiable, Valuable, Estimable, Small, and Testable (INVEST).
- Keep business rules and quality requirements as separately identified requirements.
- A story is a planning and conversation unit; its acceptance criteria carry the verifiable detail.

Use Job Stories when context and motivation explain the need better than a persona. Use Cases are preferable when a story would hide important branches or exception flows.

## 3. Modern functional and quality framework

Use **ISO/IEC 25010:2023** as the quality backbone rather than treating FURPSE as a fixed checklist. Its nine product-quality characteristics are supplemented here with privacy, observability, sustainability, and responsible-AI concerns that need explicit treatment in modern products.

### A. Functional suitability

- Feature completeness and correctness
- Business rules and calculations
- User and system workflows
- Roles, permissions, and segregation of duties
- States, transitions, idempotency, and concurrency behavior
- Inputs, outputs, validation, errors, and recovery
- Integrations, events, APIs, and interoperability
- Data lifecycle: creation, reading, update, deletion, retention, export, and lineage
- Edge cases, abuse cases, and degraded behavior

### B. Performance efficiency

- Latency percentiles, throughput, capacity, concurrency, and startup time
- Resource, network, storage, energy, and cost efficiency
- Expected load, peak load, growth assumptions, and performance degradation policy

### C. Compatibility

- Coexistence with products sharing resources or environments
- Interoperability, protocols, formats, APIs, events, and integration contracts
- Backward and forward compatibility where required

### D. Interaction capability

- Usability, learnability, discoverability, and error prevention
- Accessibility, including applicable WCAG target and assistive technology behavior
- Internationalization, localization, time zones, formats, and language
- Cross-device, responsive, offline, and low-bandwidth behavior when relevant
- User control, transparency, consent, and explainability

### E. Reliability

- Availability and service-level objectives
- Fault tolerance, retry, timeout, circuit-breaking, and graceful degradation behavior
- Data integrity and consistency
- Backup, restore, disaster recovery, RTO, and RPO
- Dependency failure and partial outage behavior

### F. Security

- Authentication, authorization, least privilege, and tenant isolation
- Confidentiality, integrity, auditability, non-repudiation, and secret handling
- Threat and abuse resistance, secure defaults, and supply-chain considerations

### G. Maintainability

- Modularity, analyzability, testability, modifiability, and reusability
- Configuration, documentation, ownership, and technical-debt constraints
- Ease and safety of diagnosis, change, validation, and release

### H. Flexibility

- Adaptability, scalability, installability, and replaceability
- Portability across required platforms, environments, regions, or providers
- API and schema versioning, upgrade behavior, backward compatibility, and deprecation policy

### I. Safety

- Prevention of harm to people, property, environment, data, or business operations
- Hazard identification, operational constraints, warnings, fail-safe states, and emergency controls
- Human oversight for consequential or irreversible actions

### J. Cross-cutting privacy and data governance

- Personal and sensitive data classification, minimization, purpose, consent, retention, deletion, and residency
- Regulatory and policy obligations

### K. Cross-cutting operability and observability

- Logs, metrics, traces, audit events, correlation, and diagnostic context
- Health signals, alert conditions, dashboards, and runbook expectations
- Supportability, administrative controls, and incident investigation
- Feature flags, progressive delivery, rollback, migration, and kill-switch behavior

### L. Cross-cutting responsible and sustainable behavior

- Fairness and harmful bias where automated decisions affect people
- Transparency, contestability, human oversight, and provenance for AI-assisted behavior
- Model quality, evaluation, hallucination, prompt injection, and fallback behavior when AI is involved
- Environmental and financial efficiency proportional to the feature's scale
- Misuse prevention and broader stakeholder impact

Do not include every category mechanically. Mark each as:

- **Applicable**: include measurable requirements;
- **Not applicable**: state a brief reason when its omission may be questioned;
- **Unknown**: add an owner and resolution action.

## 4. Writing high-quality requirements

Each requirement should be:

- necessary and tied to an objective or risk;
- singular, unambiguous, concise, and implementation-neutral;
- feasible and consistent with other requirements;
- measurable or otherwise objectively verifiable;
- assigned a priority and verification method;
- traceable to its source, parent story, or quality objective.

Use this structure:

```text
FR-001 — <title>
Requirement: The system MUST <observable behavior> when <condition>.
Rationale: <why this is needed>
Priority: Must | Should | Could | Won't now
Verification: Test | Inspection | Analysis | Demonstration
Source/trace: <objective, story, policy, risk, or stakeholder>
```

For quality requirements, use a measurable scenario:

```text
QR-PERF-001 — <title>
Context: <normal, peak, degraded, maintenance, or other environment>
Stimulus: <event or load>
Expected response: The system MUST <response>.
Measure: <threshold, percentile, duration, error budget, or other objective target>
Verification: <how and where it will be measured>
```

Avoid subjective wording unless accompanied by an agreed measurement method.

## 5. Acceptance criteria

Acceptance criteria define the observable boundary of a story or requirement.

Use Gherkin when examples clarify behavior:

```gherkin
Given <initial state and relevant permissions>
When <single action or event occurs>
Then <observable result>
And <additional observable result>
```

Include, where relevant:

- primary success;
- validation and boundary values;
- permissions and unauthorized access;
- duplicate, retry, concurrency, and idempotency behavior;
- dependency failure and recovery;
- accessibility behavior;
- analytics or audit events;
- quality thresholds.

Do not use acceptance criteria to repeat the story or prescribe internal implementation.

## 6. Prioritization and traceability

Use **Must / Should / Could / Won't now** for delivery scope. Do not use "Must" for every requirement.

Maintain a compact traceability table:

| Requirement | Supports | Priority | Acceptance criteria | Verification | Owner |
|---|---|---|---|---|---|

Every Must requirement must support an objective, policy, contractual obligation, or material risk.

## 7. Required output template

```markdown
# Feature Requirements — <feature name>

## 1. Document status
- Status:
- Owner:
- Contributors:
- Last updated:
- Target release:

## 2. Executive summary
### Problem
### Proposed outcome
### Evidence and baseline

## 3. Objectives and success measures
### Objectives
### Non-objectives
### Success metrics
### Guardrail metrics

## 4. Users, actors, and stakeholders

## 5. Scope
### In scope
### Out of scope
### Future considerations

## 6. Requirement format decision
- Selected formats:
- Rationale:

## 7. Assumptions, constraints, and dependencies

## 8. User journeys, stories, jobs, or use cases

## 9. Functional requirements

## 10. Quality requirements
### Experience and inclusion
### Performance and efficiency
### Compatibility
### Reliability and resilience
### Security
### Maintainability and flexibility
### Safety
### Privacy and data governance
### Operability and observability
### Responsible and sustainable behavior

## 11. Data, integrations, and analytics

## 12. Rollout, migration, and support

## 13. Risks and mitigations

## 14. Traceability

## 15. Open questions

## 16. Decision log
```

Omit empty optional subsections only after confirming they are not applicable.

## 8. Final quality review

Before delivering the document, verify:

- The problem and outcome are distinguishable from the proposed solution.
- Scope and non-objectives prevent predictable misunderstandings.
- The selected story or requirement formats fit the behavior being specified.
- Functional requirements cover normal, alternate, error, and permission paths.
- Applicable quality dimensions contain measurable targets.
- Security, privacy, accessibility, and operability were considered explicitly.
- Acceptance criteria are observable and testable.
- Terms are consistent and undefined acronyms are removed.
- Assumptions and unresolved decisions are visible.
- Must requirements are traceable to a genuine objective or obligation.
- No requirement silently embeds an unjustified implementation decision.

## 9. Reference anchors

Use these sources as orientation, while applying the standards and policies actually adopted by the organization:

- ISO/IEC 25010:2023 product quality model: <https://www.iso.org/standard/78176.html>
- IREB requirements engineering resources: <https://ireb.org/en/downloads>
- Agile Alliance glossary and guidance: <https://agilealliance.org/glossary/>
- W3C Web Content Accessibility Guidelines 2.2: <https://www.w3.org/TR/WCAG22/>
