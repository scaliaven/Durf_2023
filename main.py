from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
import subprocess
import pickle

    
y = []
string_lst = []

f = open("videos_data.txt", "r")
f.readline()
for i in range(1):
    content = f.readline().split(",")
    y.append(int(content[2])/int(content[1]))
    video = content[0]

    command = f"python tools/run_net.py --cfg demo/Kinetics/SLOWFAST_8x8_R50.yaml"

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the command line output
    output, error = process.communicate()      

    # Print the output result
    print("Command line output:")
    print(output.decode("utf-8"))

    if error:
        print("Error message:")
        print(error.decode("utf-8"))

f.close()

f = open("videos_data.txt", "r")
f.readline()
for i in range(1):
    content = f.readline().split(",")
    video_id = content[0].split("/")[-1]
    with open(f'{video_id}.pkl', 'rb') as video_file:
        loaded_data = pickle.load(video_file)
        action_labels = loaded_data["actions"]
        for i in range(len(action_labels)):
            action_labels[i] = action_labels[i].strip('"')
        data = " ".join(action_labels)
        string_lst.append(data)
        video_file.close()

f.close()
print(string_lst)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokens = tokenizer(string_lst, return_tensors='pt',  truncation=True, padding= 'max_length', max_length=10) # only process the fist 10 action labels of each sequence (to get output matrices of the same size)


model = DistilBertModel.from_pretrained("distilbert-base-uncased")
embedded = model(tokens["input_ids"]).last_hidden_state


# decode = tokenizer.batch_decode(sequences = tokens["input_ids"], skip_special_tokens = True)
X = embedded  # X is a numpy array consists of feature matrices (each action label is transform into the corresponding row of matrix)
X = X.detach().numpy()
X = X.reshape(X.shape[0], -1) #reshape the feature matrix of each video to a feature vector
y = np.array(y) # y is an array consists of the "popularity" of videos
kernel = 50.0**2 * RBF(length_scale=50.0)
krr = KernelRidge(alpha=1.0, kernel=kernel)
krr.fit(X, y)