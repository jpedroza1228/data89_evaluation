import smtplib
import csv
import ssl
from email.message import EmailMessage
from pyhere import here
import pandas as pd
import numpy as np
from great_tables import GT as gt

# fake name and email address list
contact = pd.read_csv(here('fake_names_emails.csv'))

if contact['name'].str.split(' ').str.len() == 2:
    contact['name'].str.split(' ')




contact[['first_name', 'last_name']] = contact['name'].str.split(' ', expand = True)
contact = contact.sort_values(['last_name', 'first_name'])
contact = contact[['last_name', 'first_name', 'email']]
gt.show(gt(contact.head()))



with open(contact, )


# --- CONFIGURATION ---
SMTP_SERVER = 'smtp.gmail.com' # Example: Gmail
SMTP_PORT = 465                # Port 465 for SSL
SENDER_EMAIL = 'your_email@gmail.com'
SENDER_PASSWORD = 'your_app_password' # NOT your login password (see notes below)
CSV_FILENAME = 'recipients.csv'

def send_bcc_email():
    # 1. Read emails from CSV
    recipients = []
    try:
        with open(CSV_FILENAME, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header row if it exists
            for row in reader:
                if row: # Check if row is not empty
                    recipients.append(row[0]) # Assuming email is in the first column
    except FileNotFoundError:
        print(f"Error: Could not find file '{CSV_FILENAME}'")
        return

    if not recipients:
        print("No recipients found in CSV.")
        return

    print(f"Found {len(recipients)} recipients.")

    # 2. Create the Email
    msg = EmailMessage()
    msg['Subject'] = "Monthly Newsletter"
    msg['From'] = SENDER_EMAIL
    
    # It is standard practice to set the 'To' field to yourself
    # so the email doesn't look like it has no recipient.
    msg['To'] = SENDER_EMAIL 
    
    # Adding the list to Bcc handles the privacy aspect
    msg['Bcc'] = recipients 
    
    msg.set_content("""
    Hello Team,

    This is an automated update sent to everyone.
    
    Best regards,
    The Management
    """)

    # 3. Send the Email
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
            print("Email sent successfully!")
            
    except smtplib.SMTPAuthenticationError:
        print("Authentication Error: Please check your email and App Password.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    send_bcc_email()