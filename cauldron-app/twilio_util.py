from twilio.rest import Client
from dotenv import load_dotenv
import os

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

def send_alert(msg: str):
    client.messages.create(
        from_='whatsapp:+14155238886',
        body=msg,
        to='whatsapp:+17328047031'
    )