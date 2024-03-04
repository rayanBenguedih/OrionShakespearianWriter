import tensorflow as tf
import shakespearianModelOrion as smo
import time
import dataBuilder as db
# Load the model

import openai
import os
import shutil

API_KEY = open("APIKEY.txt", "r").read()

client = openai.OpenAI(
    api_key=API_KEY
)



model = tf.keras.models.load_model('OrionModel.keras')

datasetBuilder = db.dataTokenizerBuilder(filename='shakespeare.txt', path_to_file='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', seq_length=100)


one_step_model = smo.singleStep(model, datasetBuilder.get_idx2char(), datasetBuilder.get_char2idx())


def mainLoop(LEN=800):
    start = time.time()
    states = None
    next_char = smo.tf.constant(['ROMEO:'])
    result = [next_char]

    for i in range(LEN):
        next_char, states = one_step_model.gen_step(next_char, states=states)
        result.append(next_char)
    result = smo.tf.strings.join(result)
    end = time.time()
    text = result[0].numpy().decode('utf-8')

    with open("shakespeare.txt", "w") as f:
        f.write(text)
    return text


text_to_send = open("shakespeare.txt", "r").read()


content = mainLoop(LEN=1000)



response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a theater narrator that will be given a text, and have to create a setting, a scene and a context for the dialogues given pretending it's shakespear text, do not include the dialogues in the answer"},
        {"role": "user", "content": text_to_send},
    ],
    model="gpt-3.5-turbo",
    )



    # Print 20 newlines
for i in range(20):
        print()

print(response.choices[0].message.content)
print("\n\nDialogue:\n\n", text_to_send)

# Wait until user presses 'Q' to exit the program
while True:
    user_input = input("Press 'Q' to exit the program: ")
    if user_input.upper() == 'Q':
        break

# Define the directory
directory = "history"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Get the list of files in the directory
files = os.listdir(directory)

# If there are no files in the directory, create a new file named tmp0
if not files:
    new_file_name = "tmp0"
else:
    # If there are files, find the highest numbered file and increment its number for the new file
    highest_num = max(int(file[3:]) for file in files if file.startswith("tmp"))
    new_file_name = f"tmp{highest_num + 1}"

# Create the new file in the directory
new_file_path = os.path.join(directory, new_file_name)
with open(new_file_path, 'w') as new_file:
    pass

# Move the content from shakespear.txt to the new file
with open("shakespeare.txt", 'r') as old_file:
    content = old_file.read()
with open(new_file_path, 'w') as new_file:
    new_file.write(content)

