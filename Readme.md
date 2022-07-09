### **Ultra-Lightweight Face Detection Application**<br><br>

**[Reference](https://github.com/onnx/models/tree/main/vision/body_analysis/ultraface)**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Input Path &nbsp;--> `input`
5. Output Path --> `output`

<br>

**Arguments**

1. `--mode | -m` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *image* or *video* or *realtime*
2. `--filename | -f` &nbsp;&nbsp;&nbsp; - Name of the image file (with extension)
3. `--downscale | -ds` - Downscale video by a factor before inference 
4. `--save | -s` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Save the processed file (`filename - Result.png`)

