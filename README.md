# PhyCRNet Better Model - Multi-Dataset Training

Enhanced Physics-Constrained Recurrent Neural Network for natural convection simulation with multi-dataset support for comprehensive training across different nanofluid materials.

## 🚀 New Features

### Multi-Dataset Training
Train PhyCRNet with randomly selected files from multiple nanofluid materials:
- **EG (Ethylene Glycol)**
- **Kerosene** 
- **n-decane**
- **Water**
- **Water-EG (50-50 mixture)**

The system automatically selects 5 random matlab files from each material's Ra (Rayleigh number) dataset, providing **25 diverse training samples** for robust model training.

## 📁 Project Structure

```
PhyCRNet_better_model_Ra_500_8.27/
├── data.py                          # Original single-file dataset loader
├── multi_dataset_loader.py          # NEW: Multi-dataset loader
├── result_naming.py                 # NEW: Unique result naming system
├── models.py                        # PhyCRNet model architecture
├── accurate_physics_loss.py         # Physics-informed loss functions
├── train_complete_physics.py        # Main training script (updated for multi-dataset)
├── create_training_plots.py         # Training visualization (updated)
├── create_complete_animation.py     # Animation generator (updated)
├── run_random_training.bat          # Windows batch script
├── run_random_training.sh           # Linux/Unix shell script
└── temp_selected_files/             # Generated directory for selected files
    └── selected_files.txt           # List of selected training files
```

## 🔧 Usage

### Quick Start (Windows)
```bash
run_random_training.bat
```

### Quick Start (Linux/Unix)
```bash
chmod +x run_random_training.sh
./run_random_training.sh
```

### Manual Steps
1. **Select random files**:
   ```bash
   python test_file_selection.py
   ```

2. **Train the model**:
   ```bash
   python train_complete_physics.py
   ```

3. **Create training plots**:
   ```bash
   python create_training_plots.py
   ```

4. **Generate animations**:
   ```bash
   python create_complete_animation.py
   ```

## 📊 Multi-Dataset Details

### File Selection Strategy
- **5 files** randomly selected from each material's Ra directory
- **Total: 25 files** providing ~33,000+ training samples
- **Diverse coverage** across different nanofluid properties

### Supported Materials & Nanofluid Properties

| Material | ν_thnf/ν_f | σ_thnf/σ_f | ρ_f/ρ_thnf | α_thnf/α_f |
|----------|------------|------------|------------|------------|
| EG       | 0.932      | 1.051      | 0.822      | 1.139      |
| Kerosene | 0.851      | 1.065      | 0.750      | 1.100      |
| n-decane | 0.834      | 1.159      | 0.735      | 1.023      |
| Water    | 0.908      | 1.050      | 0.800      | 1.155      |
| Water-EG | 0.920      | 1.050      | 0.811      | 1.150      |

## 🧮 Physics Implementation

The model implements the complete PDE system for natural convection:

1. **Continuity**: ∂U/∂X + ∂V/∂Y = 0
2. **X-momentum**: Complex momentum equation with nanofluid properties
3. **Y-momentum**: Includes buoyancy, magnetic, and porous media effects  
4. **Energy**: Heat transfer with nanofluid thermal properties

## 📈 Training Configuration

- **Epochs**: 500
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Physics Weight**: Dynamic (0.01 → 0.1)
- **Data Split**: 70% train, 20% validation, 10% test

## 🔍 Output Files

### Unique Result Naming
Each training run generates results with unique identifiers based on selected files:

**Example result directories:**
```
complete_physics_results_multi_EG_Kerosene_Water_Water-EG-50-50_n-decane_25files_604504e7/
training_plots_multi_EG_Kerosene_Water_Water-EG-50-50_n-decane_25files_604504e7/
complete_animations_multi_EG_Kerosene_Water_Water-EG-50-50_n-decane_25files_604504e7/
complete_physics_model_multi_EG_Kerosene_Water_Water-EG-50-50_n-decane_25files_604504e7_checkpoint.pth
```

### Training Results
- `complete_physics_model_[IDENTIFIER]_checkpoint.pth` - Trained model
- `complete_physics_results_[IDENTIFIER]/` - Training metrics and logs

### Visualizations  
- `training_plots_[IDENTIFIER]/comprehensive_training_analysis.png` - Training analysis
- `complete_animations_[IDENTIFIER]/` - Physics field animations

## ⚡ Performance

With multi-dataset training:
- **~33,630 total samples** from 25 diverse simulations
- **Enhanced generalization** across different nanofluid materials
- **Robust physics learning** from varied parameter ranges

## 🛠 Requirements

```
torch
numpy
scipy
matplotlib
h5py
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

## 🎯 Key Improvements

1. **Multi-Material Training**: Learn from diverse nanofluid properties
2. **Random File Selection**: Prevent overfitting to specific simulations  
3. **Automated Pipeline**: One-click training from selection to animation
4. **Enhanced Robustness**: Better generalization across material types
5. **Scalable Architecture**: Easy to add new materials/datasets
6. **🆕 Unique Result Naming**: Each training run creates uniquely named result directories
7. **🆕 Multiple Runs**: Run training multiple times with different file selections without conflicts

## 📝 Notes

- Scripts automatically handle file path differences between materials
- Fallback mechanisms ensure compatibility with single-file training
- Selected files are saved for reproducibility and debugging
- All original functionality preserved for backward compatibility