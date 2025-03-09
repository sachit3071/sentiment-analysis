import gradio as gr
import tensorflow as tf
import text_hammer as th
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForSequenceClassification.from_pretrained("Elegbede/Distilbert_FInetuned_For_Text_Classification")
# Define a function to make predictions
def predict(texts):
    # Tokenize and preprocess the new text
    new_encodings = tokenizer(texts, truncation=True, padding=True, max_length=70, return_tensors='tf')
    new_predictions = model(new_encodings)
    # Make predictions
    new_predictions = model(new_encodings)
    new_labels_pred = tf.argmax(new_predictions.logits, axis=1)
    new_labels_pred = new_labels_pred.numpy()[0]

    labels_list = ["Sadness 😭", "Joy 😂", "Love 😍", "Anger 😠", "Fear 😨", "Surprise 😲"]
    emotion = labels_list[new_labels_pred]
    return emotion

# Create a Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs=gr.outputs.Label(num_top_classes = 6),  # Corrected output type
    examples=[["Tears welled up in her eyes as she gazed at the old family photo."],
              ["Laughter filled the room as they reminisced about their adventures."], 
              ["A handwritten note awaited her on the kitchen table, a reminder of his affection."], 
              ["Harsh words were exchanged in the heated argument."],
              ["The eerie silence of the abandoned building sent shivers down her spine."],
              ["She opened the box to find a rare antique hidden inside, a total shock."]
              ],
    title="Emotion Classification",
    description="Predict the emotion associated with a text using my fine-tuned DistilBERT model."
)
# Launch the interfac
iface.launch()