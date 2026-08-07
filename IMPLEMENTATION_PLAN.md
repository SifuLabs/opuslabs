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

- [-] Detect visible faces and use their median position as a stable smart-crop
  focus. The implementation is present but its syntax/tests were not run because
  the final validation command was interrupted. Dynamic tracking and active
  speaker selection remain planned.
- [-] `smart`, `blur`, `crop`, and `fit` modes are implemented. Conversation
  and split-screen layouts remain planned.
- [-] Bold, clean, and minimal caption themes, safe positions, custom caption
  color/size, and a safe-zone brand label are implemented but not yet verified.
  Image logos and saved brand kits remain planned.
- [ ] Add transcript correction and multilingual subtitle/translation controls.

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

## Resume point

1. Run an AST parse of `main.py` and `src/clip_generator.py`. The previous
   syntax command was interrupted before returning a result.
2. Extend `tests/test_export_pipeline.py` for `smart` reframing, caption themes,
   brand labels, and the new natural-language preferences.
3. Run all Python tests and a real two-second smart-crop FFmpeg smoke render.
4. If validation passes, mark the implemented phase 4 items complete and add
   dynamic subject tracking, split-screen layouts, image logos, and brand-kit
   persistence.
5. The review workspace is under `review_workspace/`; its production build and
   rendered tests pass. Private deployment is still pending because the Sites
   create call returned no retrievable project id. Do not call create again
   until the existing site state can be discovered safely.
