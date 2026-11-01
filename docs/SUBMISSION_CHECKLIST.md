# Submission checklist

Student: Kritika Kumrawat · PRN 1272251096 · MCA E · Instructor: Rahul Inchal · Submission: 3 November 2026  
Repository: https://github.com/Kritikakumrawat/computer-vision-cca3 (branch `main`)  
Explanation video: https://drive.google.com/file/d/1KC3AiaGDgUWXWRE5pkoV7XZHZk3L9u9n/view?usp=drivesdk

The assignment asks for: original photograph, original self-recorded video,
source code pushed to Git, the Git URL in the report, the full report with
tables and measurements, the explanation video, and any files needed to run the
code. Access must be given to the instructor at
**rahul.inchal.inchal24@gmail.com** and **rahul.inchal.vf@mitwpu.edu.in**
(as stated in the assignment PDF).

## 1. Inputs and results
- [x] `inputs/original_photo.jpg` – portrait.
- [x] `inputs/original_video.mp4` – 24.7 s self-recorded walking clip (Unit 5B, Unit 5A frames).
- [x] `inputs/front_camera_video.mp4` – self-recorded front-camera clip (Unit 5A face frames, optional).
- [x] `inputs/second_photo.png` – second photograph (Unit 4); `inputs/second_photo_b.png` – same-scene photo (extra pair).
- [x] `python main.py --all`, `python -m pytest`, `python tools/capture_screenshots.py`.
- After changing any input, run in the project folder:

```
.venv\Scripts\python.exe main.py --all
.venv\Scripts\python.exe tools\capture_screenshots.py
.venv\Scripts\python.exe tools\build_report.py
.venv\Scripts\python.exe tools\export_pdf.py
```

## 2. Upload to GitHub (Command Prompt, `cmd.exe`)

`.gitignore` keeps out the personal media in `inputs/`, the virtual
environment, YOLO weights, the large output MP4, caches, `.env`/secret files,
`node_modules/` and temporary files. The report and output figures show the
student's face: the repository is currently **public**, so make it private on
GitHub (Settings → General → Danger Zone → Change visibility) if you do not want
them public, and invite the instructor as a collaborator.

```
cd /d D:\StudentComputerVisionAssignment
git config user.name "Kritika Kumrawat"
git config user.email "YOUR_GITHUB_EMAIL"
git checkout --orphan final-main
git add -A
git commit -m "CCA 3 computer vision assignment: Units 3, 4 and 5"
git branch -D main
git branch -m main
git remote add origin https://github.com/Kritikakumrawat/computer-vision-cca3.git
git push -u origin main
```

The `--orphan` steps replace the earlier local commit (authored under another
name) with one clean commit authored by you; nothing in the files changes. The
first push opens a browser window to sign in to GitHub. Afterwards:
Repository → **Settings → Collaborators → Add people** → invite the instructor.

## 3. Report
- [x] Student name, PRN, programme, instructor, date and repository URL are filled in.
- [x] Explanation-video link added (Section 11).

## 4. Explanation video (5–8 minutes)
- [x] Uploaded to Google Drive; link in the report and README.
- [ ] Make sure the Drive file is shared with the instructor (open the link in a private window to check).

## 5. Google Drive upload
Share the folder with both instructor addresses above:
- [ ] `inputs/original_photo.jpg`, `inputs/second_photo.png`, `inputs/original_video.mp4`
- [ ] `reports/CCA3_Technical_Report.pdf` (and the .docx)
- [ ] `StudentComputerVisionAssignment.zip` (code, inputs, outputs, screenshots, docs, reports)
- [ ] The explanation video

## 6. Final check
- [ ] Repository opens for the instructor.
- [ ] Drive link works for the instructor.
- [ ] Report has no remaining "PENDING" items you could complete.
