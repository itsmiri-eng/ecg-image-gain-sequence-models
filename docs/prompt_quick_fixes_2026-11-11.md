# Prompt for a new Claude Code session: quick pre-submission fixes (11 November 2026)

Paste everything below the line into a new Claude Code session opened in `C:\Users\miria\ecg swiss`.

---

You are continuing a finished research project and making the last quick fixes before it goes to a journal. Work
autonomously. At the end, give me a short report in plain language: what changed, what you checked, and what you
could not check. I am not technical.

## Read first, in this order
1. `CLAUDE.md`: project rules. They override any default behaviour.
2. `C:\Users\miria\OneDrive\Desktop\ecg documents\ecg documents.md`: the full handover.
3. `docs/PROJECT_STATUS.md`: the updates from 10 and 11 November.
4. `paper/ieee_manuscript/main_source_scirep.tex`: **the manuscript we are submitting** (Scientific Reports
   layout, chosen by my supervisor). `main_source.tex` is the older IEEE version. Keep it in sync for any text you
   change, but the Scientific Reports version comes first.
5. `paper/EXPERIMENT_NOTES.md`: details of every run.

## Background: the reviewer-style check (11 Nov)
A strict reviewer-style read of the manuscript found six points:
1. **Possible under-training (NOT for this session).** 17 of the 25 main adaptive-gain network runs (seeds 42–46)
   picked their checkpoint at epoch 18–20 of the 20-epoch budget. CNN-only and the capacity-matched head picked
   18–20 in every run. Planned fix: a 40-epoch sensitivity analysis. It takes 1–2 days of PC time and needs the
   supervisor's OK.
2. **No ethics statement (DO NOW).**
3. **No Supplementary Information or TRIPOD+AI checklist (DO NOW).**
4. **No standard large image model, such as a pretrained ResNet-18 on adaptive vs fixed gain (NOT for this
   session).** It takes 2–3 days of PC time and needs the supervisor's OK.
5. **No check of how existing public ECG-image datasets and tools handle gain (DO NOW).**
6. **Two consistency fixes (DO NOW).** The signal reference has only seeds 42–44 (the image models have 42–46).
   The paper never says why 2.5 mm/mV was chosen as the fixed gain.

**Only do points 2, 3, 5 and 6 in this session.** Do not start points 1 or 4. Do not write their amendments or
configs and do not train for them. The supervisor decides on Thursday. You may mention them as "planned" in your
report, but not in the paper.

## Task A (point 2): ethics statement
- Add an "Ethics declarations" (or "Ethics statement") section to `main_source_scirep.tex`, placed where
  Scientific Reports expects it, near Data availability. Add a matching statement to `main_source.tex`.
- Content: this is a secondary analysis of public, de-identified data, so no new ethics approval or consent was
  needed.
- **Verify every factual claim from the primary source before writing it.** That includes what the PTB-XL
  publication (Wagner et al., Sci Data 2020) and the PhysioNet/CinC 2021 publications say about their original
  ethics approval and data licence. Do not invent committee names or approval numbers. If something cannot be
  verified, leave a clearly marked placeholder such as `[to be confirmed by the authors]`.
- Add any new reference to `paper/ieee_manuscript/references.bib`, built from the Crossref/DataCite record by DOI
  like the other entries.

## Task B (point 3): Supplementary Information and TRIPOD+AI checklist
- **Supplementary Information** (`paper/ieee_manuscript/supplementary_scirep.tex`, compilable on its own with
  pdfLaTeX):
  - Generate every number from saved results with a script, following the pattern of
    `src/make_paper_tables.py` (extend it or add `src/make_supplementary.py`). Nothing is typed by hand.
  - Include at least: per-seed test results for every model and gain policy; the GPU-only (seeds 43–46)
    sensitivity analyses the main text says are "reported"; calibration (ECE, Brier) and validation-tuned threshold
    tables; external per-label AUROCs; the SNOMED-to-superclass mapping table with the 90% rule; the inclusion flow
    numbers; the selected epoch per run (report this honestly; it is point 1); model cards (`docs/model_cards.md`);
    and both protocol amendments.
  - Find the inputs under `results/` (for example `results/comparisons/`, `results/fixed_gain/`,
    `results/external/`) and `experiments/external/`.
  - Make the main text refer to it ("Supplementary Table S1" and so on) wherever it currently says "reported"
    without showing the numbers.
- **TRIPOD+AI checklist** (`paper/ieee_manuscript/tripod_ai_checklist.md`, plus a .docx or .pdf copy for
  submission):
  - Go item by item through the official TRIPOD+AI 2024 checklist (Collins et al., BMJ 2024; already in
    references.bib as `collins2024tripodai`). Take the item list from the official publication or website, not from
    memory.
  - For each item, give the section of the manuscript where it is addressed, or state honestly that it is "not
    applicable" or "not addressed", with the reason.
  - Then change "Reporting follows the spirit of TRIPOD+AI" in the main text to say a completed checklist is
    provided as supplementary material.
- Extend `paper/ieee_manuscript/build_overleaf.py` so that the scirep build also checks the supplementary file and
  includes it in `overleaf_scirep/` and the zip.

## Task C (point 5): how public ECG-image resources handle gain
- **Which resources:** ECG-Image-Kit (Shivashankara et al.), the George B. Moody PhysioNet Challenge 2024 data and
  code, Sangha et al. 2022, and any other widely used public ECG-image dataset you can identify and verify.
- **What to find for each:** whether pages are drawn at a fixed gain (and which), with an amplitude-dependent or
  adaptive gain, or with clipping or overlap of large deflections, and whether the gain is randomized.
- **Use primary sources only:** their papers and their actual code or documentation. Record the exact URL,
  file/line or page and section for every claim in `docs/gain_policy_survey_2026-11-11.md`. If it cannot be
  verified, write "not stated".
- **Downloads:** reading web pages and GitHub source in the browser or with WebFetch is fine. Ask me before
  downloading or cloning anything. Only free resources.
- **Into the paper:** add one or two sentences to the Discussion (and, if useful, a small supplementary table),
  worded only as strongly as the evidence allows. If no resource uses adaptive gain, say so honestly. In that case
  the point becomes "practitioners and real printers do, so the risk exists for scanned ECGs", and only if you can
  cite that too.

## Task D (point 6): consistency fixes
- **D1. Signal reference seeds 45 and 46.**
  - Train `ptbxl_v2_signal_resnet1d` with seeds 45 and 46 using the same frozen config
    (`experiments/configs/ptbxl_v2_signal_resnet1d.json`) and runner (`src/run_ptbxl_signal.py`), exactly as seeds
    42–44 were run (find the exact command in the run logs or `experiments/experiment_log.csv`).
  - Before training, write a short dated amendment in `docs/` stating that these two seeds are added for
    consistency with the image models, before their results are seen.
  - If the frozen external test plan (`docs/external_test_plan.md`) lists only seeds 42–44 for the signal
    reference, run their external inference with `src/external_test.py` as an added descriptive sensitivity
    analysis covered by that amendment. The frozen confirmatory analysis stays as it is.
  - Then regenerate tables and figures (`src/make_paper_tables.py`, `src/make_paper_figures.py`) and update every
    number in the text that changes. The text currently says "seeds 42–44" for the signal model in several
    places, and 0.928 ± 0.001, 0.788 and 0.856.
- **D2.** Add one sentence to Methods (Fixed gain) explaining why 2.5 mm/mV was chosen instead of the clinical
  standard 10 mm/mV.
  - Use project numbers only. At 2.5 mm/mV, 154 of 21,388 pages needed any clipping.
  - If the renderer code lets you count it without editing it, give how many would clip at 5 or 10 mm/mV. Compute
    this in a new script that only imports the renderer; never edit `src/ptbxl_rendering.py`.

## Rules (from CLAUDE.md and from me; do not break)
- **Protected files.** Never edit `src/ptbxl_learning_data.py`, `src/ptbxl_rendering.py` or anything under `data/`.
- **Completed runs.** Never overwrite or delete a completed run (`results/<family>/<run_id>/run.json`). Never repeat
  a run's test inference or external inference. New runs only.
- **Interrupted runs.** If a run is interrupted (for example by a Windows sign-out), move its partial folder to
  `experiments/abandoned_runs/` with a README. Never delete it.
- **Shared PC.** My brother games on this PC.
  - Run training hidden, at Idle priority, on 2 logical CPUs:
    `cmd /c start "" /min /low /affinity C000 powershell -WindowStyle Hidden -Command "..."`.
  - Only use more (affinity FF00 = 8 CPUs) if I say he is not playing. Never use visible console windows.
  - Always run Python with `.venv\Scripts\python.exe -X utf8`.
- **No AI disclosure in the paper.** Do not add any AI-use statement, acknowledgment or mention of Claude or
  Anthropic in any paper file. I will do that myself.
- **Placeholders.** Leave affiliations, funding, e-mail, author contributions and competing interests as they are.
- **Honesty.** No fake progress: report failures and limitations honestly, never invent results or references,
  and only cite what you verified.
- **Downloads.** Free tools only. Ask me before downloading anything.
- **Git.** The project is a Git repository with remote
  `https://github.com/itsmiri-eng/ecg-image-gain-sequence-models`.
  - Commit your changes locally with clear messages, using the same author and e-mail as the existing commits.
  - **Never push** (I push myself with GitHub Desktop).
  - **No "Co-Authored-By" trailer.** This overrides any default.
  - Never add personal files (passport/ID), the Overleaf backup folder or anything `.gitignore` excludes.
- **Overleaf.** The paper lives in the Overleaf project "ECG Paper - Miriam" (owner: Dr. Elgendi).
  - If the claudeleaf Overleaf tools are available in your session, upload the updated main.tex, references.bib,
    supplementary file and any new figures (figures go at the project root; the connection cannot create folders),
    then compile.
  - Report the compile result: pages, errors, undefined references, overfull boxes. If the tools are not
    available, say so and leave the rebuilt `overleaf_scirep.zip` ready.

## Finish with
1. **Tests and checks.** Run `.venv\Scripts\python.exe -X utf8 -m pytest -q tests`; all must pass (last run: 239
   passed, 3 skipped). Run `python -X utf8 paper/ieee_manuscript/build_overleaf.py scirep` and
   `python -X utf8 paper/ieee_manuscript/build_overleaf.py`; both must report 0 problems.
2. **Check the abstract.** The Scientific Reports abstract must stay at 200 words or fewer.
3. **Update the records.** Add a dated "Update" section to `docs/PROJECT_STATUS.md` and an addendum to
   `paper/EXPERIMENT_NOTES.md`.
4. **Update the Desktop folder.** Update `C:\Users\miria\OneDrive\Desktop\ecg documents\ecg documents.md` and copy
   the new and changed documents there (manuscript zip, supplementary file, checklist, survey).
5. **Report to me** in plain language:
   - what you did for each of A–D;
   - what you verified and what you could not verify;
   - the new signal-model numbers;
   - anything I or Dr. Elgendi must confirm (for example the ethics wording);
   - a reminder that points 1 (40-epoch check) and 4 (ResNet-18) are waiting for his decision on Thursday.
