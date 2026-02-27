# ML-Based Model for Drag and SRP

A machine learning-based model for computing atmospheric drag and Solar Radiation Pressure (SRP) on satellites using advanced ray tracing techniques.

## 🚀 Features

- **Ray Tracing Engine**: High-performance ray tracing for satellite geometry analysis
- **SRP Validation**: Tools for validating Solar Radiation Pressure models
- **Sphere Calibration**: Calibration utilities for spherical satellite models
- **Optimal Ray Tracing**: Optimized algorithms for efficient computation
- **Drag Validation**: Atmospheric drag model validation tools

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for accelerated computations

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://gitlab.com/align-nu/ml-base-model-for-drag-and-srp.git
cd ml-base-model-for-drag-and-srp
```

### 2. Create a virtual environment

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📦 Project Structure

```
ml-base-model-for-drag-and-srp/
├── core/
│   ├── monitor.py              # Visualization and monitoring tools
│   ├── optimal_ray_tracing.py  # Optimized ray tracing algorithms
│   ├── ray_tracing.py          # Core ray tracing implementation
│   ├── calibration_sphere.py   # Sphere calibration utilities
│   ├── drag_validation.py      # Drag model validation
│   ├── srp_validation.py       # SRP model validation
│   ├── ann_tools.py            # Artificial Neural Network optimization
│	└── orbit_propagation.py    # Orbit propagation with RK45
├── models/                     # 3D models directory
├── results/                    # Output results directory
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎯 Usage

### Main Files

1. Creating dataset
```
create_training_data_stl.py # Create trining data from STL
	- Input: res_x, res_y, N_SAMPLES
	- Output: Dataset - "./results/data/{MODEL_NAME}"

create_training_data_simple.py # Create training data from PyVista PolyData
	- Input: Define geometry, res_x, res_y, N_SAMPLES
	- Output: Dataset - "./results/data/{MODEL_NAME}"
```

2. Training ANN
```
optimal_ann_parallel.py # Training 
	- Input: DATA_PATH, PERTURBATION_STATE, EPOCHS
	- Output: *.txt file with weights and code implementation of the ANN
			  - ANN models - "results/optimization/{MODEL_NAME}/PERTURBATION_STATE/"
```

3. Simulation comparison
```
comparing_model_ann.py # Comparison
	- Input: mesh_path, config_path, model_name_path, n_orbits
	- Output: Simulation results - "./results/example_basic/{MODEL_NAME}/PERTURBATION_STATE/"
```


### Running SRP Validation

```bash
python ./core/srp_validation.py
```

This script validates the Solar Radiation Pressure model using ray tracing on satellite geometry.

### Running Sphere Calibration

```bash
python ./core/calibration_sphere.py
```

Calibrates the model using a spherical reference geometry for accuracy verification.

### Running Drag Validation

```bash
python ./core/drag_validation.py
```

Validates atmospheric drag computations against known models.

### Custom Ray Casting

```python
import pyvista as pv
import numpy as np
from core.optimal_ray_tracing import compute_ray_tracing_fast_optimized

# Load your satellite mesh
mesh = pv.read("models/Aqua+(B).stl")
mesh = mesh.triangulate().clean()

# Define incident direction (e.g., sun direction)
r_source = np.array([-1, 0, 0])  # X-axis direction
r_source = r_source / np.linalg.norm(r_source)

# Perform ray Casting
results = compute_ray_tracing_fast_optimized(mesh, r_source, res_x=500, res_y=500)

print(f"Hit points: {len(results['hit_points'])}")
print(f"Pixel area: {results['pixel_area']}")
```

## 🔬 Example Workflow

1. **Load satellite geometry**
2. **Define solar/velocity vector**
3. **Perform ray tracing**
4. **Compute force/torque coefficients**
5. **Validate against reference models**

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

### Code Style

This project follows PEP 8 guidelines. Format code using:

```bash
pip install black
black src/
```

## 📊 Dependencies

Key dependencies include:

- `numpy` - Numerical computations
- `pyvista` - 3D visualization and ray tracing
- `scipy` - Scientific computing utilities
- `matplotlib` - Plotting and visualization
- `rich` - Terminal output formatting

See `requirements.txt` for the complete list.

## 📝 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Elias Obreque

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👥 Authors

**Elias Obreque**  
📧 els.obrq@gmail.com

**Alice Hudson**  
📧 alicehudson09@gmail.com


## 🙏 Acknowledgments

- PyVista team for the excellent 3D visualization library
- Contributors to the scientific Python ecosystem

## 📞 Contact & Support

For questions, issues, or contributions:

- 📧 Email: els.obrq@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/ml-base-model-for-drag-and-srp/issues)
- 💡 Feature Requests: Open an issue with the `enhancement` label

## 🚧 Project Status

**Active Development** - This project is actively maintained and welcomes contributions.

### Roadmap

- [ ] GPU acceleration support
- [ ] Additional satellite geometry examples
- [ ] Web-based visualization interface
- [ ] Docker containerization
- [ ] Comprehensive test coverage

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code:
- Follows PEP 8 style guidelines
- Includes appropriate documentation
- Passes all existing tests
- Adds tests for new functionality

---

⭐ If you find this project useful, please consider giving it a star!