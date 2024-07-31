<body>
<h1>Multilayer Perceptron Project</h1>
<h2>Overview</h2>
<p>This project implements a Multilayer Perceptron (MLP) to classify breast cancer cell nuclei as malignant (M) or benign (B) based on their characteristics. The dataset used for training and validation is a CSV file containing 32 columns, with the 'diagnosis' column being the target label.</p>
<p>The neural network implementation includes the following features:</p>
<ul>
<li>At least two hidden layers by default</li>
<li>Adam optimization for training</li>
<li>Early stopping to prevent overfitting</li>
<li>Output as a probabilistic distribution using the softmax function</li>
<li>Learning curve graphs for loss and accuracy</li>
</ul>
<h2>Project Structure</h2>
<p>The project consists of three main programs:</p>
<ol>
<li><strong>Data Separation Program</strong>: Splits the dataset into training and validation sets.</li>
<li><strong>Training Program</strong>: Trains the neural network using backpropagation and gradient descent, saves the model at the end.</li>
<li><strong>Prediction Program</strong>: Loads the saved model, performs predictions on a given dataset, and evaluates using the binary cross-entropy error function.</li>
</ol>
<h2>Installation</h2>
<ol>
<li>Clone the repository:
<pre><code>git clone https://github.com/emgervais/multilayer_perceptron.git
cd multilayer_perceptron</code></pre>
</li>
<li>Install the required Python packages:
<pre><code>pip install numpy pandas scikit-learn joblib matplotlib</code></pre>
</li>
</ol>
<h2>Usage</h2>
<h3>Data Separation Program</h3>
<p>Run the following command to train the model:</p>
<pre><code>python parse.py</code></pre>
<h3>Training Program</h3>
<p>Run the following command to train the model:</p>
<pre><code>python train.py</code></pre>
<h3>Prediction Program</h3>
<p>Run the following command to make predictions:</p>
<pre><code>python predict.py</code></pre>
<h2>Learning Curves</h2>
<p>The prediction program will display learning curves for both training and validation loss and accuracy at each epoch. These graphs help visualize the performance of the model during training.</p>
<h2>Contributing</h2>
<p>Feel free to fork this repository, make improvements, and submit pull requests. Your contributions are welcome!</p>
<h2>License</h2>
<p>This project is licensed under the MIT License.</p>
</body>
