from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

save_directory = "model_pegasus"
model_name = 'google/pegasus-cnn_dailymail'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Tokenizer and model saved to:", save_directory)


# tokenizer = BartTokenizer.from_pretrained(save_directory)
# model = BartForConditionalGeneration.from_pretrained(save_directory) 

print("Loaded")
# Your summarization model or function


# def summarize_text(input_text,length):
    
#     input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
#     short = int(0.05*len(input_ids[0]))
#     medium = int(0.1* len(input_ids[0]))
#     large = int(0.15* len(input_ids[0]))

#     if (length ==1): min_length = short
#     elif (length==2): min_length=medium
#     else : min_length=large

#     summary_ids = model.generate(input_ids, max_length= int(0.2 *len( input_ids[0])), min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     print(summary)
#     return str(summary)

# input_text="""
# Virat Kohli is an Indian cricketer and former captain of the Indian national team. He is considered one of the best batsmen in the world and holds several records in cricket. He made his international debut in 2008 and has since then represented India in all three formats of the game. He has been a consistent performer for India and has been the backbone of the Indian batting line-up. In 2013, he was appointed as the vice-captain of the ODI team and became the captain of the team in 2017. Under his captaincy, India has reached several milestones including becoming the number-one team in Test cricket and reaching the final of the 2017 ICC Champions Trophy. He has also been honoured with several awards for his performances including the Sir Garfield Sobers Trophy for ICC ODI Player of the Year in 2017 and 2018. He is known for his aggressive captaincy and his never-say-die attitude on the field.
# """
# summarize_text(input_text, 1)
