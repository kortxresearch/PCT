# Release Procedure

Date created: 2026-07-10  
Last updated: 2026-07-16

Purpose: keep the journal package, repository snapshot, Zenodo/OSF archives, and GitHub remote synchronized whenever the manuscript changes.

## Trigger

Run this procedure after an author-approved manuscript or support-file change, such as an approved D5 wording patch or a post-T1b re-scoping decision. Do not rewrite `1. SUBMISSION\` merely because repository maintenance or future-work files changed.

## Steps

1. Recompile the manuscript and cover letter from the approved sources.
   - Run three LaTeX passes for `PCT_FoP_v4_submission.tex`.
   - Run three LaTeX passes for `PCT_FoP_v4_Cover_Letter.tex`.
   - Gate the release on zero LaTeX errors, zero undefined-reference warnings, and zero overfull boxes.

2. Sync `1. SUBMISSION\Foundations of Physics\`.
   - Copy the approved article `.tex` and `.pdf`.
   - Copy the approved cover-letter `.tex` and `.pdf`.
   - Copy the plain-language summary `.tex` and `.pdf` only if touched.

3. Sync `2. REPOSITORY\`.
   - Copy `PCT_FoP_v4_submission.tex` and `.pdf`.
   - Copy `PCT_FoP_v4_Cover_Letter.tex` and `.pdf`.
   - Treat `PCT_FoP_v4_submission.tex` as the canonical article source; do not generate a duplicate plain-text manuscript export for GitHub.
   - Verify the repository and journal-package article sources are byte-identical by SHA256.

4. Update repository documentation when the change affects claims, outputs, or release metadata.
   - Check `README.md`.
   - Check `MANUSCRIPT_ALIGNMENT.md`.
   - Check `README/README_EXECUTION.md` and `README/DELIVERABLES.txt`.
   - If output hashes changed, update the manuscript Appendix B.4 checksum table before compiling.

5. Rebuild the archive package.
   - Stage the repository snapshot and support files from the synchronized state.
   - Rebuild `PCT_v4_archive_update.zip`.
   - Copy the same byte-identical ZIP to `1. SUBMISSION\Zenodo\` and `1. SUBMISSION\OSF\`.
   - Verify both ZIPs have the same SHA256.
   - Update both `NEW_VERSION_NOTES.md` files with the dated change list.

6. Refresh repository checksums.
   - Run `python hash_artifacts.py` from `2. REPOSITORY`.
   - Verify `outputs\sha256SUMS.txt` matches any checksum table included in the article.

7. Append a dated entry to `MANUSCRIPT_ALIGNMENT.md`.
   - State the approved change.
   - State which files were synchronized.
   - State the article/source hash check.

8. Upload order for the author.
   - Zenodo first.
   - OSF second.
   - GitHub remote third, from the synchronized `1. SUBMISSION\GitHub\PCT\` publication payload.
   - Journal upload last, so editor-facing links resolve to v4-consistent content.

## W2 Execution On 2026-07-10

No author-approved manuscript change existed when this procedure was created. Therefore no files in `1. SUBMISSION\` were rewritten and no archive ZIP was rebuilt.

Verification performed on the existing state:

- The article source copies in `1. SUBMISSION\Foundations of Physics\` and `2. REPOSITORY\` had identical SHA256:
  `44C054489880E53BB6F00D40DEAEBA13C8DC79B01AD4C841AC45F129B8495D57`.
- `1. SUBMISSION\Foundations of Physics\PCT_FoP_v4_Cover_Letter.tex` and `2. REPOSITORY\PCT_FoP_v4_Cover_Letter.tex` had identical SHA256:
  `F697EF73DBD41665431487DFE4F8977FEEF379FA3C4F1AA2359C450FB490156C`.
- The existing article PDFs in submission and repository matched:
  `D525CD279C4ACE9EA607592EB5343B68CF810A6233E9F815289997D709610B79`.
- The existing cover-letter PDFs in submission and repository matched:
  `69AE95E733C299C09BFD2831B49C9FF55F570A6189C2CC6506E7F81A2EE13F6C`.

Compilation was performed in a temporary output directory as a release gate only, without rewriting the protected submission files. Three passes each were run for the article and cover letter. The generated logs were clean under the release gate: zero LaTeX errors, zero undefined-reference warnings, and zero overfull boxes. MiKTeX printed a general "updates not checked" warning during the run; it did not affect the LaTeX exit status or log gate.

## D5 Release Execution On 2026-07-10

Author-approved change: D5 sensitivity wording.

Executed:

- Applied the two D5 sentence replacements to the synchronized repository and journal-package article sources.
- Verified the three article source copies are byte-identical:
  `65FFB010F9766A32D768933A28C345DFB591A3C1D4004E5BC81B83D5DFB859BC`.
- Recompiled article and cover letter, three passes each, from repository sources into a temporary build directory.
- Log gate passed: zero LaTeX errors, zero undefined-reference warnings, and zero overfull boxes. MiKTeX printed its general updates-not-checked warning.
- Synchronized the compiled article and cover-letter PDFs to both `1. SUBMISSION\Foundations of Physics\` and `2. REPOSITORY\`.
- Refreshed `outputs\sha256SUMS.txt`.
- Rebuilt `PCT_v4_archive_update.zip` and copied the same byte-identical package to Zenodo and OSF folders.

## GitHub Publication Execution On 2026-07-16

- Rechecked the live `main` branch at the prior commit `12a7b97497f7cbe0896f57f9a2b187f9c51b034e`.
- Staged the complete `1. SUBMISSION\GitHub\PCT\` payload for publication, including bundled reproducibility dependencies.
- Excluded the deprecated duplicate plain-text manuscript export; `PCT_FoP_v4_submission.tex` and `.pdf` are the canonical public manuscript files.
- Removed obsolete not-executed placeholders, refreshed public documentation, and regenerated `outputs\sha256SUMS.txt`.
- Preserved exact file bytes in Git with `.gitattributes` so published files continue to match the checksum manifest.
- Set the checksum generator to emit LF on every operating system and verified the generated manifest with GNU `sha256sum --check`.
