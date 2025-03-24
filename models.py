from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim
from torch import nn


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"

        self.w = Parameter(torch.ones(1, dimensions))
        

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return torch.tensordot(x, self.w, dims=([1], [1]))
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x).item()
        return 1 if score >= 0 else -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"

            while True:
                    errores = 0
                    for batch in dataloader:
                        x = batch['x'] 
                        label = batch['label']
                        
                        pred = self.get_prediction(x)
                        
                        
                        if pred != label.item():
                           
                            self.w[0] += label.item() * x[0]
                            errores += 1
                    
                    
                    if errores == 0:
                        break



class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Arquitectura simple sin dropout
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
        Returns: a tensor of size 1 containing the loss
        """
        # Calcular la predicción
        predictions = self.forward(x)  # Pasar x a través de la red
        return mse_loss(predictions, y)  # Calcular y devolver la pérdida

    def train(self, dataset):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        
        for epoch in range(1000):
            total_loss = 0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            # Imprimir el progreso cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/1000, Loss: {total_loss / len(dataloader)}")


class DigitClassificationModel(torch.nn.Module):
    """
    Modelo de red neuronal para clasificación de dígitos en MNIST.
    """

    def __init__(self):
        super().__init__()
        input_size = 28 * 28  # 784
        output_size = 10  # 10 clases (0-9)

        # Definimos las capas sin Sequential para cumplir con las reglas del proyecto
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, output_size)  # Sin activación en la última capa

    def run(self, x):
        """
        Ejecuta la red neuronal en un batch de imágenes.

        Entrada:
            x: Tensor de tamaño (batch_size, 784)
        Salida:
            Tensor de tamaño (batch_size, 10) con los logits de cada clase.
        """
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        return self.fc4(x)  # Sin activación en la última capa

    def get_loss(self, x, y):
        """
        Calcula la pérdida de clasificación.

        Entrada:
            x: Tensor de entrada con shape (batch_size, 784)
            y: Tensor con labels en formato one-hot (batch_size, 10)
        Salida:
            Pérdida computada con CrossEntropyLoss.
        """
        return cross_entropy(self.run(x), y)

    def train(self, dataset, epochs=5, batch_size=64, learning_rate=0.001):
        """
        Entrena el modelo usando el dataset proporcionado.

        Entrada:
            dataset: Un objeto DigitClassificationDataset proporcionado por el backend.
            epochs: Número de épocas de entrenamiento.
            batch_size: Tamaño del batch para el entrenamiento.
            learning_rate: Tasa de aprendizaje.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, y = batch["x"], batch["label"]
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")




class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"

        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"

    
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """




    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """

     
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """



class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""

     