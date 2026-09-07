# Tutorial package relocation

Move OpenPinch/data/notebooks and OpenPinch/data/sample_cases into
OpenPinch/tutorials, retaining both child directory names and every file's bytes.
Runtime HPR maps and interchange contracts remain in OpenPinch/data. Public
resource helpers and sample-name loading retain their existing behavior.

This is a single package-layout refactor of the existing Python/Hatch project.
Requirements and design are fully specified by the user and existing packaging
contracts. No new business logic, user stories, infrastructure or NFR design is
needed. Existing development authorization covers implementation and verification.

- [x] Inspect resource consumers and packaging; capture original file hashes.
- [x] Relocate both packages and update resource, script, test and lint paths.
- [x] Update current public documentation for the new layout.
- [x] Verify resource identity, generators, targeted tests, strict docs and both
  installed distribution formats; record completion.

Use existing resource/copy and generator properties; confirm all moved file hashes
match. Keep historical audit records intact. PBT applies through identity and
repeatability checks; Security and Resiliency remain disabled. No commit or push
is included in this request.
