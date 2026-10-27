"""Build the technical report (reports/CCA3_Technical_Report.docx) from the generated outputs.

Every number in the report is read from the JSON/CSV files that main.py writes,
so the report always matches the latest run. Workflow:

    python main.py --all
    python tools/build_report.py
    python tools/export_pdf.py        # optional, needs Microsoft Word + pywin32

Sections whose inputs are missing (second photo, video) are marked PENDING.
"""

from __future__ import annotations

import ctypes
import importlib.metadata as md
import platform
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

from src import config, utils  # noqa: E402

OUT = config.DEFAULT_OUTPUT_DIR
REPORT_PATH = ROOT / "reports" / "CCA3_Technical_Report.docx"
ACCENT = RGBColor(0x1F, 0x3B, 0x63)
MUTED = RGBColor(0x52, 0x51, 0x4E)
GIT_URL_PLACEHOLDER = "[PENDING - paste the Git repository URL here after creating and pushing the repository]"
VIDEO_URL_PLACEHOLDER = "[PENDING - paste the Google Drive link of the explanation video here after recording and uploading it]"


def load(rel: str):
    path = OUT / rel
    return utils.load_json(path) if path.is_file() else None


# ------------------------------------------------------------------ environment
def environment_info() -> list[tuple[str, str]]:
    rows = []
    os_name = f"{platform.system()} {platform.release()}"
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                build = int(winreg.QueryValueEx(key, "CurrentBuild")[0])
                edition = winreg.QueryValueEx(key, "EditionID")[0]
            os_name = f"Windows {'11' if build >= 22000 else '10'} ({edition}), build {build}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    rows.append(("Operating system", os_name))
    rows.append(("Processor", cpu or "unknown"))
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            rows.append(("Memory (RAM visible to Windows)", f"{status.ullTotalPhys / 2**30:.1f} GB"))
    try:
        import torch

        gpu = "CUDA GPU available" if torch.cuda.is_available() else "No CUDA GPU - all inference on the CPU"
    except Exception:
        gpu = "PyTorch not importable"
    rows.append(("GPU", gpu))
    build = " (conda-forge build)" if "conda-forge" in sys.version else ""
    rows.append(("Python", f"{platform.python_version()}{build}, in the project virtual environment .venv"))
    return rows


def package_versions() -> list[tuple[str, str, str]]:
    roles = [
        ("opencv-python", "Image I/O, colour conversion, filtering, thresholding, morphology, Canny, cv2.watershed, "
                          "SIFT, BF/FLANN matching, RANSAC, Haar cascades, MOG2, Lucas-Kanade, video read/write"),
        ("numpy", "Image arrays, synthetic noise with a seeded random generator, statistics"),
        ("scikit-image", "PSNR/SSIM metrics, HOG features and visualisation, peak detection and watershed, sample data"),
        ("scipy", "Hole filling and connected-component labelling (scipy.ndimage)"),
        ("matplotlib", "All comparison figures and charts"),
        ("ultralytics", "Pretrained YOLO11n object detector (COCO, 80 classes)"),
        ("torch", "Deep-learning runtime used by Ultralytics (CPU build)"),
        ("pillow", "Writing the animated GIF preview of the motion results"),
        ("python-docx", "Generating this report"),
        ("pytest", "Automated tests"),
    ]
    rows = []
    for name, role in roles:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            version = "not installed"
        rows.append((name, version, role))
    return rows


# ------------------------------------------------------------------ document helpers
class Report:
    def __init__(self):
        self.doc = Document()
        self.figure_no = 0
        self.table_no = 0
        self._setup()

    def _setup(self):
        doc = self.doc
        section = doc.sections[0]
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin"):
            setattr(section, side, Cm(2.0))
        section.top_margin, section.bottom_margin = Cm(2.0), Cm(1.8)
        normal = doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(5)
        normal.paragraph_format.line_spacing = 1.1
        for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
            style = doc.styles[name]
            style.font.name = "Calibri"
            style.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
            style.font.size = Pt(size)
            style.font.color.rgb = ACCENT
            style.font.bold = True
            style.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
            style.paragraph_format.space_after = Pt(5)
            style.paragraph_format.keep_with_next = True
        caption = doc.styles["Caption"]
        caption.font.size = Pt(9)
        caption.font.italic = True
        caption.font.bold = False
        caption.font.color.rgb = MUTED
        zoom = doc.settings.element.find(qn("w:zoom"))
        if zoom is not None:
            zoom.set(qn("w:percent"), "100")
        # Footer with page number
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("CCA 3 Computer Vision Assignment - page ")
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED
        self._field(footer, "PAGE", size=8.5)

    @staticmethod
    def _field(paragraph, instruction: str, size: float | None = None):
        run = paragraph.add_run()
        if size:
            run.font.size = Pt(size)
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = f" {instruction} "
        sep = OxmlElement("w:fldChar")
        sep.set(qn("w:fldCharType"), "separate")
        text = OxmlElement("w:t")
        text.text = "1" if instruction == "PAGE" else "Right-click and choose Update Field to build the table of contents."
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        for element in (begin, instr, sep, text, end):
            run._r.append(element)

    def h(self, text, level=1):
        return self.doc.add_heading(text, level=level)

    def p(self, text="", bold_prefix=None, italic=False, size=None, align=None, color=None):
        paragraph = self.doc.add_paragraph()
        if bold_prefix:
            run = paragraph.add_run(bold_prefix)
            run.bold = True
        run = paragraph.add_run(text)
        run.italic = italic
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = color
        if align:
            paragraph.alignment = align
        return paragraph

    def bullet(self, text, bold_prefix=None):
        paragraph = self.doc.add_paragraph(style="List Bullet")
        if bold_prefix:
            paragraph.add_run(bold_prefix).bold = True
        paragraph.add_run(text)
        paragraph.paragraph_format.space_after = Pt(2)
        return paragraph

    def pending(self, text):
        paragraph = self.doc.add_paragraph()
        run = paragraph.add_run("PENDING: ")
        run.bold = True
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        run = paragraph.add_run(text)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return paragraph

    def placeholder(self, label, text):
        paragraph = self.doc.add_paragraph()
        paragraph.add_run(label).bold = True
        run = paragraph.add_run(text)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return paragraph

    def figure(self, rel_path, caption, width_cm=17.0):
        path = OUT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if not path.is_file():
            self.pending(f"figure not generated yet ({rel_path}).")
            return
        self.figure_no += 1
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.doc.add_paragraph(style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.add_run(f"Figure {self.figure_no}. ").bold = True
        cap.add_run(caption)
        try:
            shown = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            shown = path.name
        cap.add_run(f"  [{shown}]").font.size = Pt(7.5)

    def table(self, caption, headers, rows, widths_cm=None, font_size=8.5):
        self.table_no += 1
        cap = self.doc.add_paragraph(style="Caption")
        cap.paragraph_format.keep_with_next = True
        cap.add_run(f"Table {self.table_no}. ").bold = True
        cap.add_run(caption)
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for cell, header in zip(table.rows[0].cells, headers):
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(header))
            run.bold = True
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shading = OxmlElement("w:shd")
            shading.set(qn("w:val"), "clear")
            shading.set(qn("w:color"), "auto")
            shading.set(qn("w:fill"), "1F3B63")
            cell._tc.get_or_add_tcPr().append(shading)
        header_props = table.rows[0]._tr.get_or_add_trPr()
        repeat = OxmlElement("w:tblHeader")
        repeat.set(qn("w:val"), "true")
        header_props.append(repeat)
        for values in rows:
            cells = table.add_row().cells
            for cell, value in zip(cells, values):
                cell.text = ""
                run = cell.paragraphs[0].add_run("" if value is None else str(value))
                run.font.size = Pt(font_size)
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(1)
                    paragraph.paragraph_format.line_spacing = 1.0
        if widths_cm:
            for row in table.rows:
                for cell, width in zip(row.cells, widths_cm):
                    cell.width = Cm(width)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return table

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def toc(self):
        paragraph = self.doc.add_paragraph()
        self._field(paragraph, 'TOC \\o "1-2" \\h \\z \\u')

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.core_properties.title = "CCA 3 Computer Vision Assignment - Technical Report"
        self.doc.core_properties.subject = "Units 3, 4 and 5"
        self.doc.save(str(path))


def f2(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def find(rows, **conditions):
    return next((r for r in rows if all(r.get(k) == v for k, v in conditions.items())), None)


# ------------------------------------------------------------------ sections
def title_page(r: Report):
    for _ in range(5):
        r.p("")
    r.p("CCA 3 - Practical Coding Assignment", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, color=MUTED)
    title = r.doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Image Restoration, Segmentation,\nFeature Extraction, Object Recognition\nand Motion Estimation")
    run.bold = True
    run.font.size = Pt(21)
    run.font.color.rgb = ACCENT
    r.p("Technical report - Computer Vision, Units 3, 4 and 5", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p("Python | OpenCV | NumPy | Matplotlib | scikit-image | Ultralytics YOLO", size=10.5,
        align=WD_ALIGN_PARAGRAPH.CENTER, color=MUTED)
    for _ in range(4):
        r.p("")
    for label, value in (("Student name: ", "[Your full name]"), ("PRN / Roll number: ", "[Your PRN]"),
                         ("Programme / Division: ", "[Programme, division]"), ("Course instructor: ", "[Instructor name]"),
                         ("Submission date: ", "[Date of submission]")):
        paragraph = r.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run(label).bold = True
        run = paragraph.add_run(value)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r.p("")
    r.p(f"Results generated by the project code; report built on {date.today():%d %B %Y}.", size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER, color=MUTED)
    r.p("Yellow highlights mark details that the student must fill in or experiments that are still pending.",
        size=9, align=WD_ALIGN_PARAGRAPH.CENTER, color=MUTED, italic=True)
    r.page_break()
    r.h("Contents", 1)
    r.toc()
    r.page_break()


def objective_and_environment(r: Report, summary):
    r.h("1. Assignment objective", 1)
    r.p("The objective is to implement important techniques from image restoration, image segmentation, feature "
        "extraction, object recognition and motion estimation on the student's own photograph and self-recorded video, "
        "and to compare the techniques using measured results. The work is organised as four tasks:")
    for code, text in (("Task 1 (Unit 3): ", "image restoration (noise, Gaussian and median filtering, PSNR/SSIM) and "
                                            "morphological processing and segmentation (thresholding, erosion, dilation, "
                                            "opening, closing, Canny, watershed)."),
                       ("Task 2 (Unit 4): ", "HOG features, SIFT keypoints and descriptors, SIFT matching between two "
                                            "photographs, and matching after rotation and resizing."),
                       ("Task 3 (Unit 5): ", "object recognition with a Viola-Jones Haar cascade and a pretrained YOLO model."),
                       ("Task 4 (Unit 5): ", "video motion estimation with frame differencing, GMM background subtraction "
                                            "and Lucas-Kanade optical flow.")):
        r.bullet(text, code)

    r.h("1.1 Status of the experiments", 2)
    if summary:
        rows = [(code, t["name"], t["status"].upper(), "; ".join(t.get("pending", [])) or "-")
                for code, t in summary["tasks"].items()]
        r.table("Status of each task in the latest run (from outputs/run_summary.json).",
                ["Task", "Name", "Status", "Pending items"], rows, [1.2, 4.5, 2.2, 9.1])
    else:
        r.pending("main.py has not been run yet.")

    r.h("2. Software and hardware environment", 1)
    r.table("Environment used to produce the results.", ["Item", "Value"], environment_info(), [5, 12])
    r.table("Python libraries, installed versions and their role in the project.",
            ["Package", "Version", "Role"], package_versions(), [3, 2.2, 11.8])
    r.p("The project runs from a single entry point, main.py, which runs each unit and writes all images, figures, "
        "CSV tables and JSON result files to the outputs folder. All synthetic noise and random operations use seed "
        f"{config.RANDOM_SEED}, so repeated runs give the same numbers (timings vary slightly).")


def input_section(r: Report, r3, u4, d5, m5):
    r.h("3. Input data", 1)
    r.h("3.1 Original photograph", 2)
    photo = utils.find_input(config.DEFAULT_INPUT_DIR, "original_photo")
    if photo:
        info = r3["input"] if r3 else {}
        size_kb = photo.stat().st_size / 1024
        r.p(f"File inputs/{photo.name}: {info.get('width', '?')} x {info.get('height', '?')} pixels, "
            f"{info.get('channels', 3)} colour channels, JPEG, {size_kb:.0f} KB. The original file is kept unchanged; "
            "every result is written to the outputs folder.")
        r.p("Content: a head-and-shoulders portrait of the student in a dark navy blazer and a light striped shirt, "
            "photographed against a plain light grey-blue backdrop. It contains one person and one frontal face, large "
            "smooth regions (backdrop, blazer) and fine detail (hair strands, eyes, teeth, shirt stripes). This mix makes "
            "it suitable for noise and restoration tests, thresholding, edge detection, SIFT and face detection. It is "
            "not suitable for splitting touching objects, so a supplementary image is used for that watershed experiment "
            "(Section 4.2).")
        r.p("The copy used here has no camera metadata (EXIF), so capture device and date cannot be read from the file.",
            italic=True, size=9.5, color=MUTED)
        r.figure(photo, "Original photograph used as the clean reference image (unmodified input).", 8.5)
    else:
        r.pending("inputs/original_photo.jpg has not been supplied.")

    r.h("3.2 Second photograph (Unit 4)", 2)
    second = utils.find_input(config.DEFAULT_INPUT_DIR, "second_photo")
    if second:
        r.p(f"File inputs/{second.name}: second photograph of the same subject, used for SIFT matching.")
        r.figure(second, "Second photograph of the same subject.", 8.5)
    else:
        r.pending("inputs/second_photo.jpg (a second photograph of the same person/object/scene) has not been supplied. "
                  "The two-photo matching experiment is therefore not yet performed. A clearly labelled synthetic "
                  "tilted and re-lit copy of the original photograph is used only as a preliminary code test.")

    r.h("3.3 Original video (Unit 5)", 2)
    if m5:
        v = m5["video"]
        r.table("Properties of the self-recorded video (read by OpenCV).", ["Property", "Value"], [
            ("File", f"inputs/{v['file']}"), ("Resolution", f"{v['width']} x {v['height']}"),
            ("Frame rate", f"{v['fps']:.2f} fps" + ("" if v["fps_valid"] else " (assumed)")),
            ("Frames", v["frame_count"]), ("Duration", f"{v['duration_s']:.1f} s"),
            ("File size", f"{v['file_size_mb']:.1f} MB"), ("Processed at", m5.get("processed_size")),
        ], [5, 12])
        for warning in v.get("warnings", []):
            r.p(warning, italic=True)
        r.placeholder("Video description: ", "[Describe what the video shows: the moving person/object, the background "
                                             "and how the camera was held.]")
    else:
        r.pending("inputs/original_video.mp4 (a 15-30 s self-recorded video with a moving person or object) has not "
                  "been supplied. The motion-estimation code is complete and tested on a synthetic clip, but no results "
                  "for the student's video exist yet.")

    r.h("3.4 Supplementary data (clearly not the student's own)", 2)
    r.bullet("scikit-image's bundled 'coins' photograph (Greek coins from Pompeii, Brooklyn Museum collection, no known "
             "copyright restrictions) - used for watershed on multiple objects, because the portrait has none.",
             "Watershed sample: ")
    r.bullet("a computer-generated image of three overlapping discs, used to show how watershed splits touching "
             "objects.", "Synthetic teaching image: ")
    r.bullet("a tilted, re-lit and slightly blurred copy of the original photograph, generated by the code, used only "
             "until the second photograph is supplied.", "Preliminary matching view: ")
    r.bullet("scikit-image's public astronaut image and synthetic images/clips are used only inside the automated tests.",
             "Tests: ")


def unit3_section(r: Report, r3, s3):
    r.h("4. Unit 3 - Image restoration and segmentation", 1)
    r.h("4.1 Image restoration", 2)
    if not r3:
        r.pending("Unit 3 has not been run (python main.py --unit 3).")
        return
    p = r3["parameters"]
    rows = r3["metrics"]
    k = r3["default_kernel"]
    r.h("Method", 3)
    r.bullet(f"zero-mean Gaussian noise with standard deviation {p['gaussian_noise_sigma']:g} grey levels added to every "
             "pixel and channel, then clipped to 0-255.", "Gaussian noise: ")
    r.bullet(f"{p['salt_pepper_amount']:.0%} of pixel positions set to pure white or pure black (half each).",
             "Salt-and-pepper noise: ")
    r.bullet("cv2.GaussianBlur - a weighted average of the neighbourhood (sigma derived from the kernel size).",
             "Gaussian filter: ")
    r.bullet("cv2.medianBlur - each pixel replaced by the median of its neighbourhood.", "Median filter: ")
    r.bullet(f"kernel sizes {', '.join(str(x) for x in p['kernel_sizes'])}; figures show {k}x{k}.", "Kernels: ")
    r.bullet("PSNR = 10 log10(255^2 / MSE) in dB, and SSIM (mean structural similarity over the three colour channels), "
             "both from scikit-image, computed against the clean original at the same size and in RGB order. Higher "
             "values mean the image is closer to the original.", "Metrics: ")

    r.h("Results", 3)
    r.figure("unit3/restoration/fig_gaussian_noise_comparison.png",
             f"Gaussian noise (sigma {p['gaussian_noise_sigma']:g}) and the two {k}x{k} restorations, with PSNR/SSIM.", 12.5)
    r.figure("unit3/restoration/fig_salt_pepper_comparison.png",
             f"Salt-and-pepper noise ({p['salt_pepper_amount']:.0%}) and the two {k}x{k} restorations.", 12.5)
    for noise in ("Gaussian noise", "Salt-and-pepper noise"):
        table_rows = []
        for row in [x for x in rows if x["noise"] == noise]:
            table_rows.append((row["method"], "-" if row["kernel"] is None else f"{row['kernel']}x{row['kernel']}",
                               f2(row["psnr_db"]), f2(row["ssim"], 3), f2(row["time_ms"], 1)))
        r.table(f"{noise}: PSNR and SSIM of the noisy and restored images against the original "
                "(restoration_metrics.csv).", ["Method", "Kernel", "PSNR (dB)", "SSIM", "Time (ms)"], table_rows,
                [5.5, 2, 2.8, 2.5, 2.5])
    r.figure("unit3/restoration/fig_psnr_ssim_vs_kernel.png",
             "PSNR and SSIM against kernel size; the dashed line is the noisy input.", 15.5)
    r.figure("unit3/restoration/fig_zoomed_face_comparison.png",
             f"Zoomed face region for visual comparison ({k}x{k} kernels).")
    r.figure("unit3/restoration/fig_kernel_size_tradeoff.png",
             "Kernel-size trade-off: small, medium and large kernels on the same crop.")

    def best(noise, key):
        return r3["best"][noise][key]

    gn_noisy = find(rows, noise="Gaussian noise", method="Noisy (no filter)")
    sp_noisy = find(rows, noise="Salt-and-pepper noise", method="Noisy (no filter)")
    gb, gs = best("Gaussian noise", "by_psnr"), best("Gaussian noise", "by_ssim")
    sb = best("Salt-and-pepper noise", "by_psnr")
    sp_gauss_best = max((x for x in rows if x["noise"] == "Salt-and-pepper noise" and x["method"] == "Gaussian filter"),
                        key=lambda x: x["psnr_db"])
    g5 = find(rows, noise="Gaussian noise", method="Gaussian filter", kernel=k)
    m5 = find(rows, noise="Gaussian noise", method="Median filter", kernel=k)
    r.h("Comparison and observations", 3)
    r.table("Gaussian versus median filtering - summary of the measured results.",
            ["Noise type", "Noisy input", "Best Gaussian filter", "Best median filter", "Better technique"],
            [(noise,
              f"{f2(find(rows, noise=noise, method='Noisy (no filter)')['psnr_db'])} dB / "
              f"{f2(find(rows, noise=noise, method='Noisy (no filter)')['ssim'], 3)}",
              *[(lambda b: f"{b['kernel']}x{b['kernel']}: {f2(b['psnr_db'])} dB / {f2(b['ssim'], 3)}")(
                  max((x for x in rows if x['noise'] == noise and x['method'] == m), key=lambda x: x['psnr_db']))
                for m in ("Gaussian filter", "Median filter")],
              f"{sb['method']} (large margin)" if noise.startswith("Salt") else
              f"Similar: best PSNR {gb['method'].lower()}, best SSIM {gs['method'].lower()}"
              if gb["method"] != gs["method"] else gb["method"])
             for noise in ("Gaussian noise", "Salt-and-pepper noise")],
            [3.2, 3, 3.6, 3.6, 3.6], font_size=8)
    r.bullet(f"Salt-and-pepper noise lowered the image to {f2(sp_noisy['psnr_db'])} dB / SSIM {f2(sp_noisy['ssim'], 3)}. "
             f"The {sb['kernel']}x{sb['kernel']} median filter restored it to {f2(sb['psnr_db'])} dB / SSIM "
             f"{f2(sb['ssim'], 3)}, while the best Gaussian filter reached only {f2(sp_gauss_best['psnr_db'])} dB. "
             "A median picks one of the real neighbouring values, so isolated black and white pixels are simply "
             "discarded; a Gaussian filter averages them into their neighbours, leaving grey specks that are visible in "
             "the zoomed figure.", "Impulse noise: ")
    r.bullet("For the median filter, larger kernels lowered PSNR and SSIM on salt-and-pepper noise because the noise is "
             "already removed at 3x3 and bigger windows only remove real detail.", "Kernel size: ")
    r.bullet(f"Gaussian noise lowered the image to {f2(gn_noisy['psnr_db'])} dB / SSIM {f2(gn_noisy['ssim'], 3)}. Both "
             f"filters improved it by similar amounts: at {k}x{k} the Gaussian filter gave {f2(g5['psnr_db'])} dB and the "
             f"median {f2(m5['psnr_db'])} dB. The best PSNR was {gb['method'].lower()} {gb['kernel']}x{gb['kernel']} "
             f"({f2(gb['psnr_db'])} dB) and the best SSIM {gs['method'].lower()} {gs['kernel']}x{gs['kernel']} "
             f"({f2(gs['ssim'], 3)}).", "Gaussian noise: ")
    r.bullet("The metrics kept rising up to the largest kernel because most of this portrait is smooth (backdrop, "
             "blazer, skin), where heavy smoothing is harmless. The kernel-size figure shows the cost: large kernels "
             "also soften eyelashes, hair strands and skin texture. PSNR and SSIM summarise the whole image, so a visual "
             "check of detailed regions is still needed when choosing a kernel.", "Metrics versus visual quality: ")
    med = [x for x in rows if x["method"] == "Median filter"]
    small_t = [x["time_ms"] for x in med if x["kernel"] <= 5]
    large_t = [x["time_ms"] for x in med if x["kernel"] > 5]
    if small_t and large_t:
        r.bullet(f"the median filter took {min(small_t):.0f}-{max(small_t):.0f} ms for 3x3 and 5x5 kernels but "
                 f"{min(large_t):.0f}-{max(large_t):.0f} ms for larger kernels on this image, because OpenCV uses a fast "
                 "sorting network only for kernels up to 5x5 and a histogram-based method above that. The Gaussian "
                 "filter stayed fast at every size.", "Speed: ")

    if not s3:
        r.pending("Unit 3B results not found.")
        return
    sp = s3["parameters"]
    r.h("4.2 Morphological processing and segmentation", 2)
    r.h("Thresholding", 3)
    r.p(f"The portrait was converted to grayscale and smoothed with a {sp['blur_kernel']}x{sp['blur_kernel']} Gaussian "
        f"blur. Otsu's method chose a global threshold of {s3['otsu_threshold']:.0f}; the mask is inverted so that dark "
        "regions (hair, eyebrows, eyes, the blazer) become white foreground. Adaptive thresholding "
        f"(block {sp['adaptive_block_size']}, C = {sp['adaptive_c']}) compares each pixel with its own neighbourhood and "
        "keeps local texture and edges instead of large regions.")
    r.figure("unit3/segmentation/fig_thresholding.png", "Grayscale portrait, histogram with the Otsu threshold, and the "
                                                        "two binary masks.", 12)
    r.h("Erosion, dilation, opening and closing", 3)
    ks = sp["morph_kernel_size"]
    morph = s3["morphology"]
    r.table(f"Effect of each operation on the Otsu mask ({ks}x{ks} {sp['morph_kernel_shape']} structuring element).",
            ["Operation", "Foreground (%)", "White blobs", "Blobs < 50 px", "Enclosed holes"],
            [(m["operation"], f2(m["foreground_pct"]), m["blobs"], m["small_blobs_lt_50px"], m["holes"]) for m in morph],
            [4.5, 3, 2.8, 2.8, 2.8])
    base, ero, dil, opn, clo = morph
    r.figure("unit3/segmentation/fig_morphology.png", "Morphological operations; bottom row zooms into the hair and face.")
    r.bullet(f"erosion shrank the foreground from {f2(base['foreground_pct'])}% to {f2(ero['foreground_pct'])}% and "
             "removed thin hair strands; dilation grew it to "
             f"{f2(dil['foreground_pct'])}% and merged nearby features such as the eyes and eyebrows.", "Measured: ")
    r.bullet(f"opening reduced the number of white blobs from {base['blobs']} to {opn['blobs']} (small specks removed) "
             f"while keeping the foreground almost unchanged ({f2(opn['foreground_pct'])}%); closing reduced enclosed "
             f"holes from {base['holes']} to {clo['holes']}.", "Measured: ")
    r.bullet("opening is the better choice for removing small noise blobs, closing for filling small gaps inside "
             "regions; plain erosion and dilation change object size and are mainly useful as building blocks.",
             "Best use: ")

    r.h("Canny edge detection", 3)
    canny = s3["canny"]
    r.table("Canny edge statistics (canny_statistics.csv).", ["Input", "Low / high threshold", "Edge pixels (%)"],
            [(c["input"], f"{c['low']} / {c['high']}", f2(c["edge_pixels_pct"])) for c in canny], [8, 4, 4])
    r.figure("unit3/segmentation/fig_canny.png", "Canny edges on the clean, noisy and median-restored portrait.", 12.5)
    clean, auto, noisy, restored = canny
    r.p(f"On the clean portrait {f2(clean['edge_pixels_pct'])}% of pixels are edges (hairline, eyes, mouth, collar and "
        f"lapels). Gaussian noise raised this to {f2(noisy['edge_pixels_pct'])}% because noise creates false gradients; "
        f"a 5x5 median filter before Canny brought it back to {f2(restored['edge_pixels_pct'])}%. Restoration therefore "
        "matters for later segmentation steps. The automatic thresholds (median-based) gave "
        f"{f2(auto['edge_pixels_pct'])}% edge pixels.")

    r.h("Watershed segmentation", 3)
    ws = s3["watershed_objects"]
    r.p("Watershed treats the image like a landscape and floods it from markers; where floods from two markers meet, a "
        "boundary is drawn. For touching objects the markers are the peaks of the distance transform (the centre of "
        "each object), so two touching round objects get two markers and are separated.")
    coins = not ws.get("uses_student_photo")
    if coins:
        r.p("The portrait shows one person and no touching objects, so the touching-objects experiment uses a clearly "
            "labelled supplementary image (scikit-image 'coins', 24 coins). Uneven lighting in that photo was first "
            "removed with a morphological top-hat, then the mask was thresholded with Otsu, cleaned with opening and "
            "closing, and split with distance-transform watershed.")
    r.table("Watershed results.", ["Image", "Connected blobs after thresholding", "Markers", "Watershed regions"], [
        (ws["source"], ws["connected_components"], ws["markers_found"], ws["watershed_regions"]),
        ("Synthetic overlapping discs", s3["watershed_synthetic_touching_discs"]["connected_components"],
         s3["watershed_synthetic_touching_discs"]["markers_found"],
         s3["watershed_synthetic_touching_discs"]["watershed_regions"]),
    ], [6, 4, 2.5, 3])
    r.figure("unit3/segmentation/fig_watershed_objects.png", "Distance-transform watershed on the supplementary coins "
                                                             "image (not the student's photograph).")
    if coins:
        r.p(f"Result: the cleaned mask had {ws['connected_components']} blobs because closing joined some neighbouring "
            f"coins in the bottom row with thin bridges; the distance transform still had one peak per coin, so "
            f"watershed produced {ws['watershed_regions']} regions and split the bridged coins. The darkest coin "
            "(top-left) has low contrast, so its outline is irregular - a limitation of thresholding, not of watershed.")
    r.figure("unit3/segmentation/fig_watershed_touching_synthetic.png",
             "Synthetic teaching example: one connected blob split into three regions.", 14)
    pw = s3["watershed_portrait"]
    r.p(f"On the portrait itself, marker-based watershed (cv2.watershed) separated the person from the plain backdrop: "
        "markers came from the colour distance to the backdrop colour learned from the image border. The person mask "
        f"covers {f2(pw['person_pct'], 1)}% of the image. This is foreground/background segmentation, not separation "
        "of touching objects.")
    r.figure("unit3/segmentation/fig_watershed_portrait.png", "Marker-based watershed on the portrait.")
    r.table("Morphological operations and segmentation methods - comparison.",
            ["Technique", "What it produced on the selected image", "Works best for"], [
                ("Otsu threshold", f"Global split at {s3['otsu_threshold']:.0f}: hair, eyes and blazer as foreground",
                 "Images with two clear brightness groups"),
                ("Adaptive threshold", "Local texture and outlines", "Uneven lighting, documents, texture"),
                ("Erosion / dilation", f"Foreground {f2(ero['foreground_pct'])}% / {f2(dil['foreground_pct'])}%",
                 "Shrinking / growing shapes"),
                ("Opening / closing", f"Blobs {base['blobs']} -> {opn['blobs']}; holes {base['holes']} -> {clo['holes']}",
                 "Removing specks / filling holes"),
                ("Canny", f"{f2(clean['edge_pixels_pct'])}% edge pixels (clean)", "Thin, well-located boundaries"),
                ("Watershed (distance)", f"{ws['watershed_regions']} regions on {ws['source']}", "Touching round objects"),
                ("Watershed (markers)", f"Person mask {f2(pw['person_pct'], 1)}% of the portrait",
                 "Object vs plain background"),
            ], [3.5, 7.5, 6])


def unit4_section(r: Report, u4):
    r.h("5. Unit 4 - Feature extraction and image matching", 1)
    if not u4:
        r.pending("Unit 4 has not been run (python main.py --unit 4).")
        return
    p = u4["parameters"]
    r.h("5.1 Histogram of Oriented Gradients (HOG)", 2)
    r.p("HOG divides the image into small cells, computes the gradient direction and strength of each pixel, and "
        f"builds a {p['hog_orientations']}-bin histogram of directions per cell. Groups of "
        f"{p['hog_cells_per_block']}x{p['hog_cells_per_block']} cells (blocks) are normalised (L2-Hys) so that the "
        f"descriptor is robust to lighting. The portrait was resized to {p['hog_max_side']} px on its longer side so the "
        "cells are visible.")
    r.table("HOG feature length for different cell sizes (hog_statistics.csv).",
            ["Image", "HOG input", "Cell size", "Cells", "Feature length", "Expected length", "Time (ms)"],
            [(h["image"], h["hog_input_size"], f"{h['pixels_per_cell']} px", h["cells"], f"{h['feature_length']:,}",
              f"{h['expected_length']:,}", f2(h["time_ms"], 0)) for h in u4["hog"]],
            [2.8, 2.2, 1.8, 1.8, 2.6, 2.6, 2])
    r.figure("unit4/fig_hog.png", "HOG visualisation with 8, 16 and 32-pixel cells.", 12.5)
    hog_orig = {h["pixels_per_cell"]: h for h in u4["hog"] if h["image"] == "Original photo"}
    if 8 in hog_orig and 16 in hog_orig:
        ratio = hog_orig[8]["feature_length"] / hog_orig[16]["feature_length"]
        r.p("With 16-pixel cells the HOG picture clearly outlines the head, hairline, shoulders and lapels; 8-pixel "
            f"cells also show the eyes and mouth but produce a vector {ratio:.1f} times longer; 32-pixel cells keep only "
            "the coarse silhouette.")
    if all(h["feature_length"] == h["expected_length"] for h in u4["hog"]):
        r.p("Every measured length equals blocks across x blocks down x cells per block x orientations, which confirms "
            "the configuration.")

    r.h("5.2 SIFT keypoints and descriptors", 2)
    sift = u4["sift"]
    r.table("SIFT keypoints (sift_keypoints.csv).", ["Image", "Size", "Keypoints", "Descriptor matrix", "Time (ms)"],
            [(s["image"], s["size"], s["keypoints"], s["descriptor_shape"], f2(s["time_ms"], 0)) for s in sift],
            [4, 3, 2.5, 3.5, 2.5])
    r.figure("unit4/fig_sift_keypoints.png", "SIFT keypoints (circle size = scale, line = orientation) and the "
                                             "128-value descriptor of the strongest keypoint.")
    r.p(f"SIFT found {sift[0]['keypoints']} keypoints on the portrait. They concentrate on textured, high-contrast "
        "areas (hair, eyes, mouth, collar and shirt stripes); the smooth backdrop and the plain blazer give very few. "
        "Each keypoint has a 128-value descriptor: a 4x4 grid of sub-regions around the point, each with an 8-bin "
        "histogram of gradient directions, measured relative to the keypoint's own orientation and scale.")

    r.h("5.3 Feature matching", 2)
    r.p(f"Descriptors were matched with a brute-force matcher (L2 distance, two nearest neighbours) and Lowe's ratio test "
        f"(keep a match only if its distance is below {p['sift_ratio']} x the second-best distance). RANSAC then counted "
        "the matches that agree with one geometric model (homography for transformed copies of one image, fundamental "
        "matrix for two real photographs). For transformed copies the true transform is known, so a match is counted "
        f"as correct when it lands within {p['correct_match_tolerance']:g} px of the true position.")
    pair = u4["pair_matching"][0]
    if "PRELIMINARY" in pair["experiment"]:
        r.pending("Matching between the original and an independently captured second photograph has not been done "
                  "because inputs/second_photo.jpg is missing. The row below is a PRELIMINARY code test on a synthetic "
                  "tilted and re-lit copy of the original photograph; it is not a second photograph.")
    r.table("Matching between the two images (pair_matching.csv).",
            ["Pair", "Keypoints 1 / 2", "Ratio-test matches", "Match ratio", "RANSAC inliers", "Correct"],
            [(x["experiment"], f"{x['keypoints_1']} / {x['keypoints_2']}", x["ratio_test_matches"],
              f"{f2(x['match_ratio_pct'], 1)}%", f"{x['ransac_inliers']} ({f2(x['inlier_ratio_pct'], 1)}%)",
              f"{f2(x['correct_matches_pct'], 1)}%" if x.get("correct_matches_pct") is not None else "n/a (unknown)")
             for x in u4["pair_matching"]], [5.2, 2.4, 2.2, 2, 2.8, 2.4], font_size=8)
    match_img = "unit4/04_matches_original_vs_second.jpg" if "PRELIMINARY" not in pair["experiment"] else \
        "unit4/05_PRELIMINARY_matches_synthetic_view.jpg"
    r.figure(match_img, "Matches between the original and " + (
        "the second photograph." if "PRELIMINARY" not in pair["experiment"] else
        "the PRELIMINARY synthetic view (green = RANSAC inliers, red = rejected)."))
    mc = u4["matcher_comparison"]
    r.table("Brute-force versus FLANN matcher on the same pair (matcher_comparison.csv).",
            ["Matcher", "Ratio-test matches", "RANSAC inliers", "Matching time (ms)"],
            [(x["matcher"], x["ratio_test_matches"], x["ransac_inliers"], f2(x["match_ms"], 1)) for x in mc],
            [7, 3, 3, 3])
    bf, fl = mc
    faster = "brute force" if bf["match_ms"] < fl["match_ms"] else "FLANN"
    r.p(f"The two matchers kept {bf['ratio_test_matches']} and {fl['ratio_test_matches']} matches. Matching took "
        f"{f2(bf['match_ms'], 1)} ms with brute force and {f2(fl['match_ms'], 1)} ms with FLANN, so {faster} was faster "
        "here. FLANN is an approximate search (with an index-building cost) that pays off for large descriptor sets; "
        "with a few hundred keypoints the exact brute-force search is sufficient.")

    r.h("5.4 Matching after rotation, resizing and lighting changes", 2)
    trans = u4["transform_matching_original"]
    r.table("Original photograph versus transformed copies of itself (transform_matching_original.csv).",
            ["Transform", "Image 2 size", "Keypoints 2", "Ratio-test matches", "RANSAC inliers", "Correct (%)",
             "Median error (px)"],
            [(x["experiment"], x["image2_size"], x["keypoints_2"], x["ratio_test_matches"], x["ransac_inliers"],
              f2(x["correct_matches_pct"], 1), f2(x["median_position_error_px"], 2)) for x in trans],
            [3.6, 2.3, 1.9, 2.2, 2.2, 2, 2.4], font_size=8)
    r.figure("unit4/fig_rotation_scale_sweep.png", "Matches and correct-match rate against rotation angle and scale.",
             15.5)
    for key, cap in (("rotation", "rotated by 45 degrees"), ("resize", "resized to 0.5x"),
                     ("illumination", "darkened (gamma 2)")):
        r.figure(f"unit4/06_matches_original_vs_{key}.jpg", f"Original versus a copy {cap}.", 15)
    rot = [x for x in trans if x["transform"] == "rotation"]
    sca = [x for x in trans if x["transform"] == "resize"]
    gam = [x for x in trans if x["transform"] == "illumination"]
    rot_odd = [x for x in rot if x["value"] % 90]
    rot_right = [x for x in rot if x["value"] % 90 == 0]
    r.h("Observations", 3)
    r.bullet(f"every rotation kept at least {min(x['correct_matches_pct'] for x in rot):.0f}% correct matches, because "
             "SIFT measures each descriptor relative to the keypoint's dominant orientation. Rotations by 90 and 180 "
             f"degrees kept {min(x['ratio_test_matches'] for x in rot_right)}-{max(x['ratio_test_matches'] for x in rot_right)} "
             f"matches, but 15-45 degrees kept only {min(x['ratio_test_matches'] for x in rot_odd)}-"
             f"{max(x['ratio_test_matches'] for x in rot_odd)}: right-angle rotations move pixels exactly, while other "
             "angles need interpolation, which slightly changes local gradients.", "Rotation: ")
    small = min(sca, key=lambda x: x["value"])
    r.bullet(f"matches fell from {max(x['ratio_test_matches'] for x in sca)} to {small['ratio_test_matches']} at "
             f"{small['value']:g}x, mainly because the small image has only {small['keypoints_2']} keypoints (fine detail "
             f"disappears), but {f2(small['correct_matches_pct'], 1)}% of the remaining matches were still correct thanks "
             "to SIFT's scale-space search.", "Scale: ")
    r.bullet("; ".join(f"{x['experiment'].lower()}: {x['ratio_test_matches']} matches, "
                       f"{f2(x['correct_matches_pct'], 1)}% correct" for x in gam)
             + ". The geometry is unchanged, so the matches that survive are almost all correct, but fewer survive: "
               "descriptors are normalised gradient histograms and tolerate uniform brightness scaling, whereas a "
               "non-linear gamma change compresses contrast in the bright (gamma < 1) or dark (gamma > 1) range, so "
               "weak gradients there fall below SIFT's contrast threshold or change shape.", "Illumination: ")
    if "PRELIMINARY" in pair["experiment"]:
        r.bullet(f"the synthetic tilted and re-lit view kept {pair['ratio_test_matches']} matches, fewer than any "
                 "rotation, because a perspective change also distorts the local patches that descriptors describe. "
                 "A real second photograph adds 3-D viewpoint change, pose and expression differences and new lighting, "
                 "which SIFT handles only partially, so fewer matches and more RANSAC-rejected outliers are expected.",
                 "Viewpoint: ")
    else:
        r.bullet(f"between the two real photographs SIFT kept {pair['ratio_test_matches']} ratio-test matches, of which "
                 f"{pair['ransac_inliers']} were consistent with one epipolar geometry. Real viewpoint, pose and lighting "
                 "differences change local patches more than a pure image rotation or resize does.", "Viewpoint: ")
    second_rows = u4.get("transform_matching_second_photo") or []
    if second_rows:
        r.table("Original versus the second photograph before and after rotating/resizing the second photograph.",
                ["Experiment", "Ratio-test matches", "Match ratio", "RANSAC inliers"],
                [(x["experiment"], x["ratio_test_matches"], f"{f2(x['match_ratio_pct'], 1)}%", x["ransac_inliers"])
                 for x in second_rows], [8, 3, 3, 3])
    else:
        r.pending("Rotating/resizing the second photograph and matching it against the original (needs "
                  "inputs/second_photo.jpg). The experiment above on transformed copies of the original is complete.")


def unit5_detection_section(r: Report, d5):
    r.h("6. Unit 5 - Object recognition (Viola-Jones and YOLO)", 1)
    if not d5:
        r.pending("Unit 5 detection has not been run (python main.py --unit 5).")
        return
    haar, yolo = d5["haar"], d5["yolo"]
    r.h("6.1 Method", 2)
    r.bullet("OpenCV's pretrained haarcascade_frontalface_default.xml (and haarcascade_eye.xml inside each face). The "
             "Viola-Jones detector slides windows over an image pyramid and evaluates simple light/dark rectangle "
             "(Haar-like) features with an integral image; boosted stages reject non-faces early. Parameters: "
             f"scaleFactor {haar['parameters']['scale_factor']}, minNeighbors {haar['parameters']['min_neighbors']}, "
             f"minimum face size {haar['parameters']['min_size_fraction']:.0%} of the short image side, after histogram "
             "equalisation.", "Haar cascade: ")
    if yolo:
        r.bullet(f"Ultralytics {yolo['model']} pretrained on COCO (80 classes), run on the CPU with confidence threshold "
                 f"{yolo['confidence_threshold']}. A convolutional network predicts boxes, class labels and confidence "
                 "scores for the whole image in one forward pass.", "YOLO: ")
    else:
        r.pending(f"YOLO could not be run: {d5.get('yolo_error')}")
    r.h("6.2 Results on the portrait", 2)
    r.figure("unit5/detection/fig_haar_vs_yolo_portrait.png", "Haar cascade (amber, eyes in light blue) versus YOLO "
                                                              "(blue) on the portrait.")
    r.table("Haar cascade versus YOLO (haar_vs_yolo_comparison.csv).", ["Aspect", "Haar cascade", "YOLO"],
            [(c["aspect"], c["haar"], c["yolo"]) for c in d5["comparison"]], [3.5, 6.5, 7])
    yolo_text = (", ".join(f"{d['label']} ({f2(d['confidence'])})" for d in yolo["detections"]) or "nothing") \
        if yolo else "not run"
    r.p(f"On the upright portrait the Haar cascade returned {haar['faces']} face box(es) and {haar['eyes']} eye box(es) "
        f"in {f2(haar['time_ms'], 0)} ms; YOLO returned: {yolo_text}"
        + (f" in {f2(yolo['time_ms'], 0)} ms (both timed after one warm-up run)." if yolo else "."))
    r.h("6.3 Missed and incorrect detections (observed)", 2)
    sweep = d5["haar_min_neighbors_sweep"]
    r.table("Haar cascade - effect of minNeighbors on the portrait (one face present).",
            ["minNeighbors", "Boxes returned", "Extra (false) boxes"],
            [(s["min_neighbors"], s["faces_detected"], s["extra_detections"]) for s in sweep], [4, 4, 4])
    r.figure("unit5/detection/fig_haar_min_neighbors.png", "Haar detections for different minNeighbors values.")
    rot = d5["rotation_robustness"]
    headers = ["Rotation", "Haar faces"] + (["YOLO persons", "Best person confidence", "Other YOLO labels"] if yolo else [])
    r.table("Detections on rotated copies of the portrait (one face and one person always present).", headers,
            [(f"{x['angle_deg']} deg", x["haar_faces"]) + ((x["yolo_persons"], f2(x["yolo_best_person_conf"]),
                                                             x["yolo_other_labels"] or "-") if yolo else ())
             for x in rot], [2.2, 2.2, 2.8, 3.8, 6] if yolo else [4, 4])
    r.figure("unit5/detection/fig_rotation_robustness.png", "Haar (amber) and YOLO (blue) on rotated copies.")
    r.p("Observations recorded automatically from the measurements:", bold_prefix=None)
    for text in d5["observations"]:
        r.bullet(text)
    haar_miss = [x["angle_deg"] for x in rot if x["haar_faces"] == 0]
    if haar_miss:
        r.p(f"Why the Haar cascade fails from {min(haar_miss)} degrees: the frontal-face cascade was trained on upright "
            "faces, so its fixed rectangle features no longer line up with the eyes, nose and mouth once the face is "
            "tilted. With a low minNeighbors value it also accepts isolated windows on textured areas.")
    if yolo:
        extra = [x for x in rot if x["yolo_persons"] > 1]
        missed = [x for x in rot if x["yolo_persons"] == 0]
        text = ("YOLO learned from varied photographs and found the person at every tested angle" if not missed else
                f"YOLO missed the person at {', '.join(str(x['angle_deg']) for x in missed)} degrees")
        if extra:
            text += (f", but at {', '.join(str(x['angle_deg']) for x in extra)} degrees its confidence dropped to "
                     f"{f2(extra[0]['yolo_best_person_conf'])} and it returned an extra, overlapping person box "
                     "(a duplicate detection)")
        r.p(text + ".")
    frames = d5.get("video_frames") or []
    if frames:
        r.figure("unit5/detection/fig_video_frame_detections.png", "Haar and YOLO on frames of the original video.")
        r.table("Detections on video frames.", ["Frame", "Haar faces", "YOLO detections"],
                [(x["frame_index"], x["haar_faces"], x.get("yolo_detections", "-")) for x in frames], [3, 3, 11])
    else:
        r.pending("Detection on frames of the self-recorded video (needs inputs/original_video.mp4).")
    r.h("6.4 Face-specific cascade versus general object detector", 2)
    r.p("A Haar cascade answers a single yes/no question - 'is this window a frontal face?' - so it needs one cascade "
        "file per object type, gives no confidence score and is sensitive to rotation, pose and lighting, but it is tiny "
        "and fast on a CPU. YOLO is a general-purpose detector: one network localises and classifies 80 object classes "
        "at once with a confidence score, and is far more robust to pose and rotation, at the cost of a larger model, "
        "a deep-learning runtime and (on some machines) more computation. YOLO's COCO classes include 'person' but not "
        "'face', so for face-level tasks a face-specific model is still needed.")


def unit5_motion_section(r: Report, m5):
    r.h("7. Unit 5 - Video motion estimation", 1)
    params = config.MOTION
    r.h("7.1 Method", 2)
    r.bullet(f"frames are read with cv2.VideoCapture, resized so the longer side is at most {params['max_side']} px, and "
             "processed in order; each pair of consecutive frames is compared.", "Frame extraction: ")
    r.bullet(f"absolute difference of consecutive blurred grayscale frames, threshold {params['diff_threshold']}, small "
             "opening and dilation; contours give moving regions.", "Frame differencing: ")
    r.bullet(f"cv2.createBackgroundSubtractorMOG2 (history {params['mog2_history']}, varThreshold "
             f"{params['mog2_var_threshold']}, shadow detection on; shadow pixels are removed from the mask). Each pixel "
             "is modelled as a mixture of Gaussians; pixels that do not fit the background components are foreground.",
             "GMM background subtraction: ")
    r.bullet(f"up to {params['lk_max_corners']} Shi-Tomasi corners tracked with cv2.calcOpticalFlowPyrLK "
             f"(window {params['lk_win_size']}x{params['lk_win_size']}, {params['lk_max_level']} pyramid levels), "
             f"re-detected every {params['lk_redetect_every']} frames; points moving more than "
             f"{params['lk_moving_threshold']} px/frame are drawn as arrows.", "Lucas-Kanade optical flow: ")
    r.h("7.2 Results", 2)
    if not m5:
        r.pending("No results yet: inputs/original_video.mp4 has not been supplied. The implementation "
                  "(src/motion_estimation.py) is complete and its automated tests pass on a synthetic test clip; after "
                  "adding the video, run python main.py --unit 5 and rebuild this report.")
    else:
        r.figure("unit5/motion/fig_consecutive_frames.png", "Consecutive frames and their differences.")
        r.figure("unit5/motion/fig_motion_snapshots.png", "Foreground masks and motion vectors on selected frames.")
        r.figure("unit5/motion/fig_motion_timeseries.png", "Per-frame foreground share, Lucas-Kanade speed and "
                                                           "processing time.", 15.5)
        s = m5["summary"]
        r.table("Measured comparison over the whole video (motion_comparison.csv).",
                ["Technique", "Mean foreground (%)", "Mean regions / frame", "Mean ms / frame"],
                [(s[0]["technique"], f2(s[0]["mean_foreground_pct"]), f2(s[0]["mean_regions_per_frame"], 1),
                  f2(s[0]["mean_ms_per_frame"], 2)),
                 (s[1]["technique"], f2(s[1]["mean_foreground_pct"]), f2(s[1]["mean_regions_per_frame"], 1),
                  f2(s[1]["mean_ms_per_frame"], 2)),
                 (s[2]["technique"], f"{f2(s[2]['mean_moving_points'], 1)} of {f2(s[2]['mean_tracked_points'], 0)} "
                                     "points moving", f"{f2(s[2]['mean_moving_speed_px_per_frame'])} px/frame mean speed",
                  f2(s[2]["mean_ms_per_frame"], 2))], [6, 4, 4, 3])
        r.p(f"Agreement: the frame-differencing and MOG2 masks overlapped with a mean IoU of {f2(m5['mean_mask_iou'], 3)}, "
            f"and {f2(m5['mean_lk_moving_in_gmm_pct'], 1)}% of the moving Lucas-Kanade points fell inside the MOG2 "
            "foreground.")
        r.placeholder("Observations: ", "[Describe what you see in the figures: where each method marks the moving "
                                        "person/object, false motion from camera shake or lighting, and holes in the masks.]")
    r.table("Characteristics of the three techniques (general properties, not measurements).",
            ["Property", "Frame differencing", "GMM (MOG2)", "Lucas-Kanade"], [
                ("Output", "Changed pixels between 2 frames", "Pixels unlike the learned background",
                 "Motion vector per tracked point"),
                ("Memory of the past", "Previous frame only", "Statistical model of many frames", "Previous frame only"),
                ("Typical weakness", "Hollow masks for uniform objects; misses objects that stop",
                 "Needs warm-up; absorbs objects that stay still; lighting changes", "Sparse; loses points on fast "
                                                                                       "motion or occlusion"),
                ("Camera motion", "Whole frame marked as moving", "Whole frame marked as foreground",
                 "Measures it (global flow)"),
                ("Gives direction/speed", "No", "No", "Yes"),
            ], [3.2, 4.4, 4.7, 4.7], font_size=8)


def limitations_and_conclusion(r: Report, r3, s3, u4, d5, m5):
    r.h("8. Limitations and pending work", 1)
    items = []
    if not utils.find_input(config.DEFAULT_INPUT_DIR, "second_photo"):
        items.append("Two-photo SIFT matching and the rotation/resizing of the second photo are pending until "
                     "inputs/second_photo.jpg is added; the synthetic view is only a preliminary test.")
    if not m5:
        items.append("All motion-estimation results and detections on video frames are pending until "
                     "inputs/original_video.mp4 (15-30 s, self-recorded) is added.")
    for text in items:
        r.pending(text)
    r.bullet("PSNR and SSIM measure similarity to the reference image, not perceived quality; the visual comparisons "
             "should be read together with the numbers.")
    r.bullet("The coins image for watershed is a supplementary public image; adding your own photo of touching objects "
             "as inputs/watershed_photo.jpg makes the experiment use your own data automatically.")
    r.bullet("Rotation, scaling and lighting experiments use transformed copies of one photograph, which isolates each "
             "effect but does not reproduce real 3-D viewpoint changes.")
    r.bullet("Timings were measured on a laptop CPU (no GPU) and vary between runs.")

    r.h("9. Conclusion", 1)
    if r3:
        sb = r3["best"]["Salt-and-pepper noise"]["by_psnr"]
        gauss_rows = [x for x in r3["metrics"] if x["noise"] == "Gaussian noise" and x["kernel"] is not None]
        best_g = max((x for x in gauss_rows if x["method"] == "Gaussian filter"), key=lambda x: x["psnr_db"])
        best_m = max((x for x in gauss_rows if x["method"] == "Median filter"), key=lambda x: x["psnr_db"])
        gap = abs(best_g["psnr_db"] - best_m["psnr_db"])
        gauss_text = (f"the Gaussian and median filters performed almost equally (best PSNR {f2(best_g['psnr_db'])} vs "
                      f"{f2(best_m['psnr_db'])} dB)" if gap < 1.0 else
                      f"the {max(best_g, best_m, key=lambda x: x['psnr_db'])['method'].lower()} performed better")
        r.p(f"Restoration: the {sb['method'].lower()} is the best choice for salt-and-pepper noise ({f2(sb['psnr_db'])} dB "
            f"with a {sb['kernel']}x{sb['kernel']} kernel). For Gaussian noise {gauss_text}, and the kernel size must "
            "balance noise removal against loss of detail.")
    if s3:
        r.p("Segmentation: Otsu thresholding separated the dark hair and blazer from the light background and skin; "
            "opening and closing cleaned the mask; Canny showed that noise must be filtered before edge detection; "
            "watershed split touching objects using distance-transform markers and separated the person from the "
            "backdrop with colour-based markers.")
    if u4:
        rot = [x for x in u4["transform_matching_original"] if x["transform"] == "rotation"]
        r.p(f"Features: HOG captured the overall shape of the portrait, and SIFT matching stayed accurate under rotation "
            f"(at least {min(x['correct_matches_pct'] for x in rot):.0f}% correct matches), scaling and lighting "
            "changes; the number of matches drops mainly when the image becomes small.")
    if d5:
        rot = d5["rotation_robustness"]
        haar_miss = [x["angle_deg"] for x in rot if x["haar_faces"] == 0]
        false_haar = [s["min_neighbors"] for s in d5["haar_min_neighbors_sweep"] if s["extra_detections"] > 0]
        parts = [f"the Haar cascade found {d5['haar']['faces']} face on the upright portrait"]
        if haar_miss:
            parts.append(f"missed it at {', '.join(str(a) for a in haar_miss)} degrees of rotation")
        if false_haar:
            parts.append(f"produced false boxes with minNeighbors {', '.join(str(n) for n in false_haar)}")
        text = "Recognition: " + ", ".join(parts) + "."
        if d5.get("yolo"):
            dup = [x["angle_deg"] for x in rot if x.get("yolo_persons", 0) > 1]
            miss = [x["angle_deg"] for x in rot if x.get("yolo_persons", 1) == 0]
            text += " YOLO " + ("found the person at every angle" if not miss else
                                f"missed the person at {', '.join(str(a) for a in miss)} degrees")
            text += (f" but returned a duplicate box at {', '.join(str(a) for a in dup)} degrees." if dup else ".")
        r.p(text)
    if m5:
        r.p("Motion: see Section 7 for the measured comparison of the three techniques on the self-recorded video.")
    else:
        r.p("Motion estimation is implemented and tested, and its results will be added once the self-recorded video "
            "is supplied.")

    r.h("10. Git repository and explanation video", 1)
    r.placeholder("Git repository URL: ", GIT_URL_PLACEHOLDER)
    r.placeholder("Project explanation video (Google Drive): ", VIDEO_URL_PLACEHOLDER)
    r.p("Access for the instructor must be granted on both the repository and the Google Drive folder (see "
        "docs/SUBMISSION_CHECKLIST.md).", italic=True, size=9.5)


def appendix(r: Report):
    r.h("Appendix A - Source code", 1)
    r.p("The complete source code is in the project folder (and the Git repository). Each task is one module:")
    r.table("Source files.", ["File", "Contents"], [
        ("main.py", "Command-line entry point: --unit 3/4/5, --all, --check-inputs, --skip-yolo"),
        ("src/config.py", "All file names and parameters (noise levels, kernel sizes, thresholds, model name)"),
        ("src/utils.py", "Input validation, image I/O, figures, JSON/CSV writing, timer"),
        ("src/image_restoration.py", "Noise models, Gaussian/median filters, PSNR/SSIM, restoration figures"),
        ("src/segmentation.py", "Otsu/adaptive thresholding, morphology, Canny, watershed (objects and portrait)"),
        ("src/feature_extraction.py", "HOG, SIFT, BF/FLANN matching, RANSAC, rotation/scale/lighting experiments"),
        ("src/object_detection.py", "Haar cascade face/eye detection, YOLO detection, robustness experiments"),
        ("src/motion_estimation.py", "Video validation, frame differencing, MOG2, Lucas-Kanade, output video/GIF"),
        ("tests/", "Automated tests (pytest) on synthetic and public sample data"),
        ("tools/build_report.py", "Builds this report from the outputs"),
    ], [5, 12])
    r.h("Appendix B - How to reproduce", 1)
    for line in ("python -m venv .venv", r".venv\Scripts\activate", "pip install -r requirements.txt",
                 "python main.py --all", "python -m pytest", "python tools/build_report.py"):
        paragraph = r.doc.add_paragraph()
        run = paragraph.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        paragraph.paragraph_format.space_after = Pt(0)


def build(path: Path = REPORT_PATH) -> Path:
    r3 = load("unit3/restoration/restoration_results.json")
    s3 = load("unit3/segmentation/segmentation_results.json")
    u4 = load("unit4/unit4_results.json")
    d5 = load("unit5/detection/detection_results.json")
    m5 = load("unit5/motion/motion_results.json")
    summary = load("run_summary.json")
    report = Report()
    title_page(report)
    objective_and_environment(report, summary)
    input_section(report, r3, u4, d5, m5)
    unit3_section(report, r3, s3)
    unit4_section(report, u4)
    unit5_detection_section(report, d5)
    unit5_motion_section(report, m5)
    limitations_and_conclusion(report, r3, s3, u4, d5, m5)
    appendix(report)
    report.save(path)
    return path


if __name__ == "__main__":
    out = build()
    print(f"Report written to {out}")
