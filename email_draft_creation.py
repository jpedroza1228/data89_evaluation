import pandas as pd
# pip install pyhere | uv pip install pyhere | conda install pyhere
from pyhere import here

mastery_df_path = ''
stu_name_path = ''
email_df = ''

attr_mastery = pd.read_csv(here(f'{mastery_df_path}.csv'))
names_df = pd.read_csv(here(f'{stu_name_path}.csv'))

# need to make sure that students are sorted alphabetically to make sure proficiency values match student names
attr_mastery = attr_mastery.join(names_df)

def email_template(row):
  name = row['name']
  email = f"Hi {name}, \n\nYou are receiving this email because you were flagged as a student that may not be proficient in one or more of the concepts that were being assessed on this week's quiz. We are recommending that you spend some time reviewing the course material pertaining to the concepts you may not be proficient in that were covered in this week's quiz. \n\nAs a reminder, probabilistic models are not perfect. If you believe this model is not an accurate representation of your quiz performance, you may dismiss this email. If you believe this assessment accurately represents your understanding of the topics covered on the quiz, there are additional resources that you may want to utilize. Resources such as homework parties and office hours, as well as unofficial resources like studying with your peers may be helpful for further developing your understanding of the course material. \n\nBased on our model, our assessment of your proficiency is as follows: \n\n1: {row['attr1']}\n2: {row['attr2']}\n3: {row['attr3']}\n4: {row['attr4']}\n5: {row['attr5']}"
  return email

attr_mastery['email_draft'] = attr_mastery.apply(email_template, axis = 1)

# attr_mastery.to_csv(f'{email_df}.csv')