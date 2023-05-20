This repository contains solutions for Assignment-3. 


## **Objective:** 
To build a transilteration system, our aim is to generate the Telugu word that corresponds to a given romanized word. For example, when provided with the word ```prayaanikulu ```our system will produce the Telugu word ```ప్రయాణికులు```.


## **Dataset**: 
Sample Dataset is provided by **AI4Bharat**.
* Number of samples in train: 51.2K
* Number of samples in valid: ~4.1k
* Number fo samples in test :  ~4.1K

## **Folder structure:**

* **WandB_Sweep_Vanilla.ipynb:**  Contains the code to experiment with the different hyparparemeters and observe the results in WandB. We don't have attention mechanism here. 
* **WandB_Sweep_Attention.ipynb.ipynb:** Contains the code to experiment with the different hyparparemeters with Attention support.
* **Best_Model_Vanilla.ipynb:** Model will be built based on the best configuration (for vanilla models) and inference is done on the test data set. 
* **Best_Model_Attention.ipynb:** Model will be built based on the best configuration (for Attention models) and inference is done on the test data set and attention scores are visualaized t understand the which character is given more wegihtage at each of the time step.
* **utilities.py:** Conatins all the helper functions/classes. 
* **train.py**: Single Command line script which supports traning the model with/without Attention.



## **How to use:**
* run the command 
  ``` python train.py``` > It trains a model with best configuaration and prints test accuracy. 

* One can overwrite the default values by passing command line aruguments.


### **Arguments supported:**
<br>


| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | CS22M080 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | CS22M080 | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
 `-attn`, `--attention` | Yes | To use attention mechanism or not |
| `-e`, `--cell_type` | GRU |  Specify which cell type to use.|
| `-e`, `--epochs` | 15 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 256 | Batch size used to train neural network. |
| `-el`, `--encoder_num_layers` | 3 | Number of encoder layers |
| `-dl`, `--decoder_num_layers` | 2 | Number of decoder layers |
| `-hs`, `--hidden_size` | 512 | cell hidden state size|
| `-es`, `--embedding_size` | 256 | embedding size|
| `-do`, `--dropout` | 0.2 | Dropout used in in cells.|
| `-bi`, `--bidirectional` | Yes | To use Bi directional cell or not.|

<br>
