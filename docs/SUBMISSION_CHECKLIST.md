# Submission checklist

The assignment asks for: original photo, original video, source code pushed to
Git, the Git URL in the report, the full report with tables and measurements, the
explanation video, and any files needed to run the code. Access must be given to
the instructor at **rahul.inchal.inchal24@gmail.com** and
**rahul.inchal.vf@mitwpu.edu.in** (as stated in the assignment PDF).

## 1. Finish the inputs and results
- [ ] Add `inputs/second_photo.jpg` (same subject, your own camera) and
      `inputs/original_video.mp4` (15–30 s, self-recorded) – see `inputs/README.md`.
- [ ] `python main.py --all` – no task should say FAILED.
- [ ] `python -m pytest` – all tests pass.
- [ ] Look at the new figures in `outputs/unit4` and `outputs/unit5/motion` and
      write your observations into the highlighted placeholder in Section 7.2 of the report.

## 2. Git repository (private, then invite the instructor)

The local repository is already initialised with one commit (see `git log`).
Personal media in `inputs/`, the virtual environment, YOLO weights and the large
output video are excluded by `.gitignore`. The figures in `outputs/` show your
portrait, so keep the repository **private**.

1. On github.com: **New repository** → name e.g. `CCA3-Computer-Vision` →
   **Private** → do *not* add a README/.gitignore (the project has them) → Create.
2. In PowerShell, in the project folder:

```
git add -A
git commit -m "Add second photo and video results"     # only if you changed something
git remote add origin https://github.com/<your-username>/CCA3-Computer-Vision.git
git branch -M main
git push -u origin main
```

   The first push opens a browser window to sign in to GitHub (Git Credential Manager).
3. Repository → **Settings → Collaborators → Add people** → invite the instructor
   (GitHub accepts an e-mail address or a GitHub username).
4. Copy the repository URL into the report (Section 10, *Git repository URL*).

If the instructor needs a public repository instead, decide first whether you are
comfortable publishing figures that show your face.

## 3. Report
- [ ] `python tools/build_report.py` then `python tools/export_pdf.py`.
- [ ] Open `reports/CCA3_Technical_Report.docx` in Word, fill in the yellow fields
      (name, PRN, programme, instructor, date, Git URL, video link, video
      description, motion observations), remove the highlighting, update the table
      of contents (right-click → Update Field) and save as PDF again.
      (Re-running `build_report.py` overwrites manual edits, so do this last.)

## 4. Explanation video
- [ ] Record 5–8 minutes in your own words following `docs/VIDEO_SCRIPT.md`.
- [ ] Upload to the instructor's Google Drive folder and paste the link into the report.

## 5. Google Drive upload
Upload to the Drive folder provided by the instructor and share it with both
addresses above:
- [ ] `inputs/original_photo.jpg`, `inputs/second_photo.jpg`, `inputs/original_video.mp4`
- [ ] `reports/CCA3_Technical_Report.pdf` (and the .docx)
- [ ] `StudentComputerVisionAssignment.zip` (code, outputs, docs – personal inputs not included)
- [ ] The explanation video

## 6. Final check
- [ ] Repository opens for the instructor (test with a private/incognito window
      after they accept, or ask them).
- [ ] Drive link works for the instructor.
- [ ] Report contains both links and no remaining "PENDING" items you could complete.
