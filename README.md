# Task Analysis - Image Processing for Board Game Tracking

## Task 1: Detecting the Last Added Piece

The first task requires finding the position of the last piece added to the board in the current image. To accomplish this, we use three auxiliary scripts:

### Auxiliary Scripts:

#### `stilizare.py`
- Applies transformations to the image to facilitate contour extraction.
- Converts the image to **HSV format** to apply specific filters, extracting relevant colors into a mask.
- Applies **blur** followed by **Canny Edge Detection** to retain only prominent contours above a certain threshold.
- The result highlights marked board squares (e.g., x3 multipliers).

#### `crop.py`
- Uses the preprocessed image from `stilizare.py` to extract the game board.
- Detects contours in the image and draws bounding rectangles around them to ensure detected objects have straight edges.
- Identifies the **top-left and bottom-right corners** to crop the image accurately.

#### `diferenta.py`
- Compares the current image with the previous one to identify the exact position of the newly added piece.
- The key difference between images is the newly placed piece.
- Divides each image into **14x14 grid cells** and iterates through them to find the indices where the maximum difference occurs.

### `task1.py`
The process starts by applying all transformations to an **empty board** as a baseline for detecting the first added piece. Then, we iterate through board images, applying the three auxiliary functions while also performing additional transformations:

- Apply a filter similar to `stilizare.py` but optimized for highlighting game pieces.
- Resize the image to ensure consistency, which is necessary for `diferenta.py` to perform accurate subtraction.
- Save the indices **(i, j)** where the maximum difference is detected into a text file with the current image's name.

---

## Task 2: Identifying the Added Piece

This task builds upon Task 1 by adding an auxiliary script to generate **templates of game pieces**.

### `extragere.py`
- Extracts all pieces from an image containing all available pieces.
- Saves each piece as an individual file, named based on the number present in its square.
- Applies transformations to isolate the piece's number as the only visible information before saving it.

### Process
1. Detect the square where the change occurred (from Task 1).
2. Apply similar filters as in `extragere.py` to the extracted square.
3. Use **`matchTemplate()`** to compare the extracted square with stored templates.
4. Identify the closest match and append the piece's value to the text file created in Task 1.

---

## Task 3: Score Calculation and Game Tracking

This task involves tracking scores and player moves while integrating previous tasks.

### Steps:
1. **Create tracking files** to store:
   - Current player
   - Number of pieces added per turn
   - Points per turn
2. **Process move history** from the text files generated in Task 1 and 2.
   - Store moves as a **list of lists** for better management.
   - Maintain a **round counter** to track game progression.
3. **Compare the round counter with each player's local maximum.**
   - If exceeded, remove the player from the list, save the necessary data, and reset their score.
4. **Use Task 2's implementation to track moves in a matrix.**
   - Check adjacent squares for **two numbers**.
   - Determine if any of the **four arithmetic operations** (addition, subtraction, multiplication, division) result in the number on the current square.
   - Update the score accordingly.
5. **Game Reset Condition**:
   - When the counter reaches **50**, all game values reset.

---
## Summary
This project leverages image processing to track and analyze board game state changes in real-time. It utilizes **image segmentation, template matching, and numerical computation** to detect new pieces, identify them, and calculate game scores dynamically.

### Key Technologies Used
- **OpenCV** for image processing
- **NumPy** for numerical operations
- **Python** for scripting and automation

This modular approach ensures flexibility, allowing for easy adjustments and enhancements to the detection and tracking system.

