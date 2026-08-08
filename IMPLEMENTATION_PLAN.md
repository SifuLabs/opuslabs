# OpusLabs Improvement Plan

This file is the implementation ledger for the product improvements agreed on
on 2026-08-07. Work is completed in dependency order so later product features
are built on a reliable video-processing foundation.

## Status legend

- `[ ]` Planned
- `[-]` In progress
- `[x]` Completed and verified
- `[!]` Requires external credentials, platform approval, or user validation

## 1. Export reliability and complete output packages

- [x] Honor caption enable/disable settings in every renderer.
- [x] Add real `blur`, `crop`, and `fit` vertical reframing modes.
- [x] Generate a valid preview thumbnail for every rendered clip.
- [x] Apply current per-platform export constraints.
- [x] Export subtitles and a machine-readable clip manifest.
- [x] Verify with automated tests and a real FFmpeg smoke render.

Acceptance check: one processing request produces playable clips plus preview
images, subtitle files, and JSON/CSV metadata without claiming unavailable
features.

## 2. Application reliability and maintainability

- [x] Consolidate the duplicated `main.py` and `main_production.py` entrypoints.
- [x] Require a real transcription path before reporting full processing.
- [x] Migrate Gemini calls from the legacy SDK to `google-genai`.
- [x] Use schema-constrained AI responses and validate timestamps.
- [x] Add unit and end-to-end tests for the processing pipeline.

Acceptance check: the CLI has one canonical entrypoint, reports capabilities
accurately, and passes repeatable offline tests.

## 3. Review workspace

- [x] Add review-batch creation and upload progress.
- [x] Show generated candidates with playable previews.
- [x] Let users edit in/out points, titles, captions, and export options.
- [x] Persist explicit approval decisions before export or publishing.

Acceptance check: a user can complete the core workflow without editing files
or using a terminal.

## 4. Smart reframing and brand controls

- [x] Detect visible faces, stabilize their sampled positions, and interpolate
  a moving smart-crop focus throughout each clip. The renderer safely falls
  back to a centered crop when OpenCV or a detectable face is unavailable.
  Audio-visual active-speaker selection remains a future enhancement.
- [x] Add `smart`, `blur`, `crop`, `fit`, and two-speaker `split`/conversation
  layouts and verify their FFmpeg filter graphs with real renders.
- [x] Add bold, clean, and minimal caption themes, safe positions, custom
  caption color/size, text labels, raster image logos, and persistent named
  brand kits.
- [x] Apply direct or JSON-file transcript corrections before clip selection,
  and export optional translated SRT sidecars while retaining source timings.
- [!] Live translation requires `google-genai` and Gemini credentials. The
  schema-constrained translation path and SRT output are covered offline, but
  an API call was intentionally not made during local validation.

Acceptance check: users can keep important subjects visible and reuse a
consistent visual identity across exports.

## 5. Projects and batch jobs

- [ ] Persist projects and processing state.
- [ ] Use isolated working directories per job.
- [ ] Add queues, progress, cancellation, retry, and resume.
- [ ] Support processing several source videos in one project.

Acceptance check: interrupted work can resume safely and concurrent jobs do not
overwrite one another.

## 6. Publishing integrations

- [ ] Define a provider-neutral publishing interface.
- [ ] Add OAuth account connections and secure token storage.
- [ ] Add draft upload first, then direct publishing and scheduling.
- [ ] Validate privacy, metadata, and platform-specific requirements.

Acceptance check: supported platforms can receive an approved clip as a draft.
Live integrations require the relevant developer applications and credentials.

## 7. Analytics feedback loop

- [ ] Import views, engaged views, retention, shares, and conversions.
- [ ] Compare predicted engagement with actual results.
- [ ] Surface per-style and per-platform recommendations.
- [ ] Use historical outcomes to rerank future clip candidates.

Acceptance check: recommendations cite observed project performance instead of
only generic heuristics.

## Change log

- 2026-08-07: Created the ordered implementation ledger and began phase 1.
- 2026-08-07: Completed the phase 1 implementation; validation remains.
- 2026-08-07: Verified phase 1 with four unit tests and a real FFmpeg render.
- 2026-08-07: Consolidated the app entrypoint, migrated Gemini, tightened
  capability checks, and added timestamp validation tests.
- 2026-08-07: Built and validated the durable review workspace with clip
  uploads, playback, timing/copy edits, filters, and approval decisions.
- 2026-08-07: Private Sites publishing paused because site creation returned no
  project record; the validated local source was preserved without retrying.
- 2026-08-07: Began phase 4. Added face-sampled smart-crop focus, caption themes,
  safe caption positioning, custom colors/sizes, and text brand labels. Work was
  paused before syntax and regression validation at the user's request.
- 2026-08-08: Completed phase 4 implementation. Repaired a manifest-export
  regression, added dependency-light and dynamic face tracking, split layouts,
  image logos, persistent brand kits, transcript corrections, and translated
  subtitle sidecars. Sixteen tests and real smart/split/logo FFmpeg renders pass.

## Resume point

1. Validate face tracking against representative real footage after installing
   OpenCV, and validate one translated subtitle request with the configured
   Gemini account. Add active-speaker selection only after that footage set is
   available for measuring subject-switch accuracy.
2. Begin phase 5 with a persisted project/job model and isolated workspaces;
   this is the dependency for reliable queues, cancellation, retry, and resume.
3. The review workspace is under `review_workspace/`; its production build and
   rendered tests pass. Private deployment is still pending because the Sites
   create call returned no retrievable project id. Do not call create again
   until the existing site state can be discovered safely.
