# Notebooks

The assignment allows either Python source files or Jupyter notebooks. This
project uses plain Python modules (`src/`) plus `main.py`, which are easier to
test and to run on any machine.

If you prefer to demonstrate the code in Jupyter (for example in the
explanation video), install Jupyter into the same virtual environment and
start a notebook in the project folder:

```
pip install notebook
jupyter notebook
```

Then, in a notebook cell:

```python
%cd ..
%run main.py --unit 3
from IPython.display import Image
Image("outputs/unit3/restoration/fig_salt_pepper_comparison.png")
```

You can also call individual functions, for example:

```python
from src import utils, image_restoration as ir
img = utils.read_image("inputs/original_photo.jpg")
noisy = ir.add_salt_and_pepper_noise(img, amount=0.05)
print(ir.compare_with_reference(img, ir.median_filter(noisy, 3)))
```
