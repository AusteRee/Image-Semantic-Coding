# RL-based-semantic-coding

This repository provides the official implementation of the paper:

**"Reinforcement Learning-Based Layered Lossy Image Semantic Coding"**



## 1. Test

To test the model, follow these steps:

1. Download the pre-trained weight file from the following Google Drive link:

   [Pre-trained Weights](https://drive.google.com/uc?export=download&id=1vjv4-J-PEEjoriWibgcLZ1rHIzq8Nlke)

2. Place the downloaded weight file in the `checkpoints/` directory.

3. Run the test script:

   ```bash
   python test.py
   ```

## Full Workflow

To run the full pipeline from semantic map to final encoded image:

1. **Generate reconstructed images:**
   - Follow the instructions in `semantic_image_synthesis/README.md` to generate reconstructed images from semantic maps.

2. **Run the semantic coding framework:**
   - Copy the generated images to `datasets/synthesized_image`.
   - Run:

    ```bash
    python train.py
    ```

The framework will perform reinforcement learning-based quantization and save the results in `Image Semantic Coding/output/`.

