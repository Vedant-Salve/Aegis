## License Plate Redaction

### Overview
The License Plate Redaction module automatically detects and anonymizes vehicle license plates in images to protect vehicle privacy and prevent identification.

### Features
- **Multiple Redaction Methods**:
  - **Blur**: Applies Gaussian blur to make plates unreadable
  - **Pixelate**: Creates a mosaic effect on detected plates
  - **Blackout**: Completely blacks out plate regions

- **Flexible Processing**:
  - Single image processing
  - Batch directory processing
  - Adjustable redaction strength
  - Detection visualization for debugging

### Technical Details
- **Detection Engine**: OpenCV Haar Cascade Classifier
- **Supported Formats**: JPG, JPEG, PNG, BMP
- **Detection Method**: Multi-scale detection with tunable parameters

### Usage

#### Command Line Interface
```bash
python app.py
# Select option 4 for License Plate Redaction
```

#### Python API
```python
from Aegis.plate_redactor import LicensePlateRedactor

# Initialize with preferred method
redactor = LicensePlateRedactor(method='blur')

# Process single image
num_plates = redactor.process_image(
    'input.jpg',
    'output.jpg',
    strength=51
)

# Process directory
processed, total_plates = redactor.process_directory(
    'input_folder/',
    'output_folder/',
    strength=51
)
```

#### Jupyter Notebook
See `Notebooks/redact_plates.ipynb` for interactive examples and visualizations.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | 'blur' | Redaction method: 'blur', 'pixelate', or 'black' |
| `strength` | int | 51 | Blur strength (must be odd) or pixel size for pixelation |
| `show_detections` | bool | False | Draw green boxes around detected plates |

### Examples

**Blur with default settings:**
```python
redactor = LicensePlateRedactor(method='blur')
processed, count = redactor.redact(image, strength=51)
```

**Pixelation with custom pixel size:**
```python
redactor = LicensePlateRedactor(method='pixelate')
processed, count = redactor.redact(image, strength=15)
```

**Complete blackout:**
```python
redactor = LicensePlateRedactor(method='black')
processed, count = redactor.redact(image)
```

### Integration with Full Pipeline

The license plate redaction seamlessly integrates with other Aegis modules:

```python
def full_anonymization(image_path, output_path):
    image = cv2.imread(image_path)
    
    # 1. Redact license plates
    plate_redactor = LicensePlateRedactor()
    image, _ = plate_redactor.redact(image)
    
    # 2. Redact faces
    # face_redactor.redact(image)
    
    # 3. Redact PII text
    # pii_redactor.redact(image)
    
    # 4. Apply adversarial cloaking
    # fawkes_cloak(image)
    
    cv2.imwrite(output_path, image)
```

### Performance
- Processing time: ~50-200ms per image (depending on resolution)
- Detection accuracy: Good for standard rectangular plates
- Works best with: Clear, front-facing plate images

### Limitations
- May miss heavily angled or partially occluded plates
- Performance depends on image quality and lighting
- Designed for standard rectangular license plates

### Future Improvements
- Deep learning-based detection (YOLO)
- Support for non-standard plate formats
- OCR-based detection for better accuracy
- Multi-region plate format support