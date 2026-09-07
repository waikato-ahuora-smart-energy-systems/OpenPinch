# Complete test-suite repair

Continue the authorized verification workflow. Run pytest without test-name or
marker exclusions, enabling all optional tutorial profiles. Keep the repository's
explicitly skipped benchmark marked as such. Preserve saved notebook evidence.

- [x] Reproduce the two notebook-state failures and identify the contract mismatch.
- [x] Correct saved-versus-generated notebook validation and run affected checks.
- [x] Run the complete suite with all tutorial profiles and fresh branch coverage;
  fix any additional failures and rerun affected tests.
- [x] Verify lint/format and coverage, then record final results and skip reasons.

PBT remains enabled; retained review properties and byte-preservation checks
complement Jupyter schema validation. Other extensions remain disabled. Existing
approval covers repairs and verification; no new planning or deployment stage.
