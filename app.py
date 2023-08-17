from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

save_directory = "models"

# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

tokenizer = BartTokenizer.from_pretrained(save_directory)
model = BartForConditionalGeneration.from_pretrained(save_directory) 

print("Loaded")
# Your summarization model or function



app = Flask(__name__,template_folder="template")
CORS(app)

def summarize_text(input_text,length):
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    short = int(0.12*len(input_ids[0]))
    medium = int(0.2* len(input_ids[0]))
    large = int(0.3* len(input_ids[0]))

    if (length ==1):
        min_length = short
        max_length = int(0.2*(len(input_ids[0])))
    elif (length==2): 
        min_length=medium
        max_length = int(0.3*(len(input_ids[0])))
    else : 
        min_length=large
        max_length = int(0.4*(len(input_ids[0])))

    summary_ids = model.generate(input_ids, max_length= max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(summary)
    return str(summary)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_summarizer.html')
def text_summarizer():
    return render_template('text_summarizer.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        input_text = request.form.get('input_text')
        slider_value = int(request.form.get('slider_value'))

        print("Value Received" + " *"*50)
        summary = summarize_text(input_text, slider_value)
        print("Done")
        return jsonify({'summary': summary})
    except Exception as e:
        # Print the exception traceback to the server logs for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    app.run(debug=False)