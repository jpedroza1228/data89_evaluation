import pandas as pd
# pip install pyhere | uv pip install pyhere | conda install pyhere
from pyhere import here


attr_mastery = pd.read_csv(here('student_data/attr_mastery_quiz1.csv'))

# attr_mastery = attr_mastery.merge(names_df, 'inner', on = 'anon_id')

def email_template(row):
    email = f"""
Hi {{{{name}}}},\n

You are receiving this email because you were flagged as a student that may not be proficient in one or more of the concepts that were being assessed on this week's quiz. We are recommending that you spend some time reviewing the course material pertaining to the concepts you may not be proficient in that were covered in this week's quiz.\n

As a reminder, probabilistic models are not perfect. If you believe this model is not an accurate representation of your quiz performance, you may dismiss this email. If you believe this assessment accurately represents your understanding of the topics covered on the quiz, there are additional resources that you may want to utilize. Resources such as homework parties and office hours, as well as unofficial resources like studying with your peers may be helpful for further developing your understanding of the course material.\n

Based on our model, our assessment of your proficiency is as follows:\n

1: {row['attr1']}
2: {row['attr2']}
"""
    return email

# Apply the function to your DataFrame
attr_mastery['email_draft'] = attr_mastery.apply(email_template, axis=1)

not_proficient = attr_mastery.loc[attr_mastery[['attr1', 'attr2']].apply(lambda x: x.str.contains('not', case = False, na = False)).any(axis = 1)]

print(attr_mastery.loc[0, 'email_draft'])

not_proficient.to_csv(here('student_data/quiz1_email_list.csv'))