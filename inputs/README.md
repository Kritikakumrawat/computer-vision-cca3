# Input files

Put your own media in this folder with exactly these names. The original files
are never modified; every result goes to `outputs/`.

| File | Needed for | Status when this project was delivered |
|---|---|---|
| `original_photo.jpg` (or `.jpeg` / `.png`) | Units 3, 4, 5A | Present - the portrait supplied by the student |
| `second_photo.jpg` (or `.jpeg` / `.png`) | Unit 4 two-photo matching | Present - `second_photo.png`, close-up photo supplied by the student |
| `second_photo_b.jpg` (optional, or `.png`) | Unit 4 extra same-scene pair | Present - `second_photo_b.png`, wider photo of the same scene |
| `original_video.mp4` (or `.mov` / `.avi` / `.mkv`) | Unit 5B motion estimation, Unit 5A video frames | Present - 24.7 s clip recorded by the student (received via WhatsApp) |
| `front_camera_video.mp4` (optional) | Unit 5A face detection on 5 frames | Present - 5.9 min front-camera clip recorded by the student |
| `second_image_video_frame.jpg` (optional) | Unit 4 fallback only when `second_photo` is missing (made by `tools/extract_video_frame.py`, always labelled as a video frame) | Not used - a second photograph is present |
| `watershed_photo.jpg` (optional) | Unit 3 watershed on your own photo | Optional - a documented scikit-image sample is used if absent |

## What to capture

- **Second photo** - the *same* person, object or scene as `original_photo.jpg`,
  taken again with your own phone/camera from a slightly different position,
  distance or lighting. Do not use an unrelated or downloaded image.
- **Original video** - 15-30 seconds, recorded by you, with a moving person or
  object and enough background detail. Keep the phone still (or on a support) for
  at least part of the clip. Do not edit or trim it.
- **Watershed photo (optional)** - a few coins, buttons or sweets touching each
  other on a plain table, photographed from above.

## After adding files

```
python main.py --check-inputs
python main.py --all
python tools/capture_screenshots.py
python tools/build_report.py
python tools/export_pdf.py
```

## Privacy

`.gitignore` keeps everything in this folder except this README out of Git.
Upload the original photo and video to the Google Drive folder given by the
instructor (the assignment asks for large raw videos to go to Drive, not Git).
