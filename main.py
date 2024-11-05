"""
This script demonstrates the usage of the Binoculars class for detecting AI-generated text.
It initializes the Binoculars model and uses it to compute a score and make a prediction for a sample text.
"""

from binoculars import Binoculars

bino = Binoculars()

# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"
sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

# Compute the Binoculars score for the sample text
print(bino.compute_score(sample_string))  # 0.75661373

# Predict whether the sample text is AI-generated or human-generated
print(bino.predict(sample_string))  # 'Most likely AI-Generated'
