filename = "./my_lstm_with_markov_weights.h5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics='accuracy')

input, output = make_rhymes("i know people that love chris rock")

# Example sentence
s1 = "The quick brown fox jumps over the lazy dog"
s2 = "Yeah so what you wanted block"

p1 = calculate_perplexity(s1)
p2 = calculate_perplexity(s2)
print(f"Perplexity of the sentence 1 \"{s1}\" is: {p1}")
print(f"Perplexity of the sentence 2 \"{s2}\" is: {p1}")

# Example sentence
s3 = "I love you"

p3 = calculate_perplexity(s3)

print(f"Perplexity of the sentence 3 \"{s3}\" is: {p3}")

import textstat

text = """
this is outrageous
"""

# Flesch-Kincaid Grade Level
fk_grade = textstat.flesch_kincaid_grade(text)
print(f"Flesch-Kincaid Grade Level: {fk_grade}")

# Gunning Fog Index
gunning_fog = textstat.gunning_fog(text)
print(f"Gunning Fog Index: {gunning_fog}")


