from joblib import load

text = 'I love my girl, she is so stunning.'

pipeline = load("mnb_model.joblib")

print(pipeline.predict([text]))

