# DTvsRFvsXG
Decision Tree Vs Random Forest Vs XGBoost
Cracking the Code: How Decision Trees, Random Forests, and XGBoost Decode Handwritten Digits
Imagine teaching a computer to read your handwriting - specifically, to recognize digits like 0 to 9 scrawled on a page. Sounds like magic, right? But with machine learning (ML), it's just a matter of training models to spot patterns in pixels. In this article, we'll dive into a fun experiment: comparing three ML models - Decision Tree, Random Forest, and XGBoost - to classify handwritten digits from the MNIST dataset. Think of these models as detectives solving a mystery, each with their own style of cracking the case. By the end, you'll see which model shines, how they work under the hood, and why this matters for learning (and teaching!) ML. Plus, we'll use colorful visualizations to make it as clear as a sunny day!
Sample Handwritten NumbersWhat's the MNIST Puzzle?
The MNIST dataset is like a giant coloring book with 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels. That's 784 tiny squares per image, with shades from black (0) to white (255). Our goal? Train ML models to look at these pixel patterns and guess the correct digit. It's a classic ML challenge, perfect for beginners and pros alike, and a great way to explore how different models tackle the same problem.
To set the stage, we preprocess the data:
Normalize: Divide pixel values by 255 to squeeze them between 0 and 1, making it easier for models to learn.
Split: Use 60,000 images for training and 10,000 for testing.

Now, let's meet our three detectives!
Meet the Detectives: Our ML Models
Each model approaches the MNIST mystery differently, like detectives with unique skills. Here's a quick intro, explained so even a kid could get it:
Decision Tree: Think of this as a curious kid playing "20 Questions." It asks yes-or-no questions about pixels (e.g., "Is Pixel 352 bright?") and builds a flowchart to guess the digit. Simple but sometimes gets stuck on tricky cases.
Setup: We use entropy for decisions, limit the tree to 20 levels (max_depth=20), and require at least 5 samples to split (min_samples_split=5).

Random Forest: Imagine a team of 100 Decision Trees voting together, like a jury. Each tree looks at a random subset of pixels, and their combined guess is usually spot-on. It's like crowd wisdom!
Setup: 100 trees (n_estimators=100), max depth of 15, and sqrt features per split. We also use out-of-bag (OOB) scoring to check performance during training.

XGBoost: This is the master detective who learns from mistakes. It builds trees one by one, each correcting the previous one's errors, like a student acing a test by studying past quizzes.
Setup: 150 trees (n_estimators=150), max depth of 10, learning rate of 0.1, and 80% subsampling for robustness.

We coded these in Python using scikit-learn and xgboost, running everything in a Jupyter notebook or Google Colab. Here's how we set it up:
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost
This installed all the tools we need, like numpy for math, matplotlib and seaborn for plots, and our ML libraries.
The Investigation: Training and Testing
We trained each model on the 60,000 training images and tested them on the 10,000 test images. Our key question: How accurately can each detective identify digits? We also wanted to peek into their methods using visualizations, like:
Confusion Matrix: A heatmap showing which digits they mix up (e.g., mistaking 3 for 8).
Feature Importance: A chart of the most important pixels for guessing digits.
Predictions: A gallery of test images with each model's guesses, marked right or wrong.

Let's see how our detectives performed!
Case Closed: Results and Insights
Accuracy Scores
After running the models, here's how they scored on the test set:
Random Forest and XGBoost cracked the case with over 96% accuracy, while Decision Tree lagged at 88.5%. Why? The solo Decision Tree is like a single detective who can miss clues, while Random Forest's team and XGBoost's error-correcting approach catch more details.
Confusion Matrices
The confusion matrices tell us where the models got confused:
Decision Tree: Often mixed up 3 and 8 or 4 and 9, because their shapes look similar (curvy loops or angled lines).

Confusion Matrix for Decision Tree Classifier

Random Forest: Fewer mistakes, but still tripped up on 8 vs. 3 occasionally.

Confusion Matrix for Random Forest Classifier

XGBoost: The sharpest detective, with minor errors on 7 vs. 1 (probably due to slanted strokes).
It's like trying to tell apart two similar-looking friends - ensemble models are better at noticing tiny differences!

Confusion Matrix for XGBoost ClassifierFeature Importance
Which pixels are the VIPs (Very Important Pixels)? The feature importance plots show:
Decision Tree: Focused on central pixels but scattered its attention, like a kid doodling randomly.

Random Forest: Picked pixels that form digit strokes, like the curve of a 2 or loop of an 8.

XGBoost: Similar to Random Forest but gave extra weight to edge pixels, sharpening its guesses.
For kids, think of it like finding the key dots that make a number's shape pop out, like stars forming a constellation!

Prediction Gallery
We picked five test images and checked each model's guesses:
Example: A true "3" was mislabeled as "8" by Decision Tree (oops!), but Random Forest and XGBoost got it right.

Pattern: Decision Tree stumbles on blurry or unusual digits, while the others are more confident.

Why This Matters (Especially for Teaching!)
This experiment isn't just about numbers - it's about understanding how ML models think. Here's why it's cool:
Beginner-Friendly: Decision Tree is like a simple game of questions, perfect for introducing ML to newbies or even kids. Random Forest and XGBoost show how teamwork (or learning from mistakes) makes models stronger.
Visual Learning: The plots are like storyboards, showing how models solve the puzzle. They're great for classrooms, where you can say, "Look, the computer picks these pixels to guess a 5!"
Real-World Impact: Recognizing digits is used in apps like check-scanning or postal sorting. Understanding these models helps you build smarter tech.

As someone who loves teaching ML (like my plan to teach Pandas in 45 minutes!), I find these visualizations super helpful for explaining complex ideas simply. For example, tell students, "Random Forest is like a class voting on the answer - more votes, better guess!

Tips to Make Your Own ML Adventure
Want to try this yourself? Here's a quick guide:
Get the Data: Load MNIST using sklearn.datasets.load_digits or tensorflow.keras.datasets.mnist.
Code the Models: Use this snippet to start:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
digits = load_digits()
X, y = digits.data / 16.0, digits.target  # Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt = DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
xgb = XGBClassifier(n_estimators=150, max_depth=10, learning_rate=0.1, random_state=42)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Test accuracy
print("Decision Tree:", accuracy_score(y_test, dt.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, rf.predict(X_test)))
print("XGBoost:", accuracy_score(y_test, xgb.predict(X_test)))
3. Visualize: Use seaborn for confusion matrices and matplotlib for feature importance and predictions. Check my GitHub repo (link below) for full code!
https://github.com/Pavan-Aditya-Kumar-Gorrela/DTvsRFvsXG
4. Experiment: Tweak parameters like max_depth or n_estimators to see how accuracy changes.

Wrapping Up: Which Detective Wins?
XGBoost takes the crown with 97.2% accuracy, closely followed by Random Forest at 96.8%. Decision Tree, at 88.5%, is simpler but less reliable. For teaching, Decision Tree's clarity is great, but Random Forest and XGBoost show the power of teamwork and learning from mistakes. The visualizations - confusion matrices, feature importance, learning curves, and predictions - make it easy to explain ML to anyone, from kids to classmates.
Want to dive deeper? Check out my code and plots on GitHub or try tweaking the models yourself. And if you're teaching ML, use these visuals to spark curiosity - nothing says "ML is cool" like a computer decoding a scribbled 8!
Have you tried ML on MNIST or another dataset? Share your story in the comments, and let's keep the ML adventure going!
