from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")


def translator(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", padding=True)
    outputs = model.generate(input_ids)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_text


# text in English
# texts = [
#     "I spend a few hours a day maintaining my website.",
#     "Where do random thoughts come from?",
#     "I can't believe that she is older than my mother.",
#     "My Mum tries to be cool by saying that she likes all the same things that I do",
#     "A song can make or ruin a person’s day if they let it get to them.",
# ]

with open("./data/fearofthedark.txt", "r") as f_in:
    with open("./data/fearofthedark_hindi.txt", "a") as f_out:
        lines = f_in.readlines()
        for line in lines:
            text = line.strip()
            if text:
                # print("English Text: ", text)
                # print("Hindi Translation: ", translator(text))
                # print("*" * 50, "\n")
                f_out.write(translator(text) + "\n")
            else:
                f_out.write("\n")
