# Damage Claim Verification Evaluation Report

This file is both a submission template and the default output target for:

```powershell
python evaluation/evaluate.py --labels sample_claims.csv --predictions output.csv --report evaluation/evaluation_report.md
```

When the evaluator is run, measured metrics and error analysis will replace the placeholder sections below.

## 1. Dataset summary

| Item | Value |
|---|---:|
| Labeled rows in `sample_claims.csv` | TBD |
| Prediction rows in `output.csv` | TBD |
| Matched rows evaluated | TBD |
| Missing predictions | TBD |
| Extra predictions | TBD |
| Total image count | TBD |
| Average images per claim | TBD |

## 2. Metrics

| Metric | Accuracy |
|---|---:|
| Claim status accuracy | TBD |
| Issue type accuracy | TBD |
| Object part accuracy | TBD |
| Severity accuracy | TBD |

## 3. Error analysis

| Field | Common mismatch | Example count | Notes |
|---|---|---:|---|
| `claim_status` | TBD | TBD | TBD |
| `issue_type` | TBD | TBD | TBD |
| `object_part` | TBD | TBD | TBD |
| `severity` | TBD | TBD | TBD |

## 4. Common failure modes

- TBD: poor lighting, blur, glare, obstruction, or incorrect angle.
- TBD: claimed part not clearly visible.
- TBD: multiple damages in the user conversation but only one visible in evidence.
- TBD: ambiguous evidence requirements.
- TBD: subtle damage that is difficult to distinguish from normal wear.

## 5. Improvement recommendations

- Add stricter image capture guidance for each object and issue type.
- Add more examples for subtle scratches, cracks, stains, and packaging damage.
- Expand `evidence_requirements.csv` with normalized `requirement_type` values.
- Use manual review routing for high-risk but visually inconclusive cases.
- Track per-image failure reasons to improve data collection.

## 6. Operational analysis

### Approximate model calls

For each successfully processed claim:

| Stage | Model call? | Approximate calls |
|---|---:|---:|
| Claim extraction | Yes | 1 |
| Vision analysis | Yes | 1 |
| Evidence check | No | 0 |
| Risk assessment | No | 0 |
| Final decision | No | 0 |

Planning formula:

```text
model_calls ≈ number_of_claims * 2
worst_case_agent_calls ≈ number_of_claims * 2 * agent_attempts
```

The default pipeline uses application-level retries plus OpenAI SDK retries, so transient failures may increase total requests.

### Token estimates

These are planning estimates and should be replaced with measured usage if available:

```text
claim_extraction ≈ 1,500 input tokens + 250 output tokens per claim
vision_analysis_text ≈ 3,000 input tokens + 600 output tokens per claim
image_tokens_or_image_cost = model-dependent
```

For `N` claims:

```text
estimated_text_input_tokens ≈ N * 4,500
estimated_text_output_tokens ≈ N * 850
```

### Image counts

```text
total_images = sum(image_paths per row)
max_images_per_claim = 8
billable_images_or_image_tokens = model-dependent after resizing/detail settings
```

The project preprocesses images with Pillow and caps dimensions to reduce unnecessary cost and latency.

### Pricing assumptions

Pricing changes over time. Before final submission, verify current official OpenAI pricing and fill in:

| Assumption | Value |
|---|---:|
| Claim model | `gpt-4.1-mini` by default |
| Vision model | `gpt-4.1-mini` by default |
| Input price per 1M tokens | TBD |
| Output price per 1M tokens | TBD |
| Image pricing basis | TBD |
| Image price estimate | TBD |

Cost formula:

```text
estimated_cost =
  (input_tokens / 1,000,000 * input_price_per_1m_tokens)
  + (output_tokens / 1,000,000 * output_price_per_1m_tokens)
  + image_processing_cost
```

### Estimated cost

| Scenario | Claims | Images | Estimated cost |
|---|---:|---:|---:|
| Small sample | TBD | TBD | TBD |
| Hackathon test run | TBD | TBD | TBD |
| Full dataset | TBD | TBD | TBD |

### Latency estimates

Planning estimates:

| Stage | Estimated latency |
|---|---:|
| Claim extraction | 1–3 seconds |
| Vision analysis | 4–12 seconds |
| Evidence/risk/decision logic | <100 ms |
| Sequential total per claim | 5–15 seconds |

Throughput can improve with `--workers`, but concurrency should be increased carefully to avoid rate limits.

### Batching strategy

- Default batch size: 10 rows.
- The system writes checkpointed output after each batch.
- Resume is enabled by default.
- Existing completed rows in `output.csv` are skipped unless `--no-resume` is used.
- Batching is local checkpoint batching, not the OpenAI Batch API.

### Retry strategy

- Claim extraction and vision analysis use application-level retries.
- The OpenAI SDK also has retry support through `--sdk-retries`.
- Backoff includes jitter.
- Failed rows are converted to safe `not_enough_information` decisions with `manual_review_required` risk context.

### Rate-limit considerations

- Start with `--workers 1`.
- Increase gradually only after confirming stable throughput.
- If rate limits occur, reduce `--workers`, reduce image detail/size, or rerun with resume.
- Keep batch checkpoints small enough to avoid losing progress.
- For larger production workloads, request higher API limits or move to an asynchronous batch design.
